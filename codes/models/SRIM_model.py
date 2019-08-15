import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss


class SRIMModel(BaseModel):
    def __init__(self, opt):
        super(SRIMModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                print('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                print('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # add extra pixel loss
            if 'pixel_weight_1' in train_opt:
                l_pix_type = train_opt['pixel_criterion_1']
                if l_pix_type == 'l1':
                    self.cri_pix_1 = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix_1 = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w_1 = train_opt['pixel_weight_1']
            else:
                # print('Remove pixel loss.')
                self.cri_pix_1 = None

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_HR=True, code=None):
        # LR
        self.var_L = data['LR'].to(self.device)

        # Code input
        if code is None:
            self.code = None
        else:
            self.code = code.to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def get_loss(self):
        result = torch.empty(self.var_L.shape[0])
        for i in range(self.var_L.shape[0]):
            gen_img = self.netG(self.var_L[i: i + 1], self.code[i: i + 1])
            l_g_total = 0
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(gen_img, self.var_H[i: i + 1])
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H[i: i + 1]).detach()
                fake_fea = self.netF(gen_img)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            if self.cri_pix_1:  # pixel loss
                l_g_pix_1 = self.l_pix_w_1 * self.cri_pix_1(gen_img, self.var_H[i: i + 1])
                l_g_total += l_g_pix_1
            result[i] = l_g_total.detach().cpu()
        return result

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.code)

        l_g_total = 0
        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
            l_g_total += l_g_pix
        if self.cri_fea:  # feature loss
            real_fea = self.netF(self.var_H).detach()
            fake_fea = self.netF(self.fake_H)
            l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea
        if self.cri_pix_1:  # pixel loss 1
            l_g_pix_1 = self.l_pix_w_1 * self.cri_pix_1(self.fake_H, self.var_H)
            l_g_total += l_g_pix_1

        l_g_total.backward()
        self.optimizer_G.step()

        # set log
        # G
        if self.cri_pix:
            self.log_dict['l_g_pix'] = l_g_pix.item()
        if self.cri_fea:
            self.log_dict['l_g_fea'] = l_g_fea.item()
        if self.cri_pix_1:
            self.log_dict['l_g_pix_1'] = l_g_pix_1.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, self.code)
        self.netG.train()

    def get_features(self):
        self.netG.eval()
        out_dict = OrderedDict()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, self.code)
        out_dict['gen_feat'] = self.netF(self.fake_H).detach().double().cpu()
        out_dict['real_feat'] = self.netF(self.var_H).detach().double().cpu()
        self.netG.train()
        return out_dict

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                print('Number of parameters in F: {:,d}'.format(n))
                message = '\n\n\n-------------- Perceptual Network --------------\n' + s + '\n'
                with open(network_path, 'a') as f:
                    f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            if 'load_partial' in self.opt and self.opt['load_partial']:
                exception = None if 'exception' not in self.opt else self.opt['exception']
                self.load_partial_network(load_path_G, self.netG, exception=exception)
            elif 'load_partial_cat' in self.opt and self.opt['load_partial_cat']:
                code_channel = 5 if 'out_code_nc' not in self.opt['network_G'] else self.opt['network_G']['out_code_nc']
                self.load_partial_network_concat(load_path_G, self.netG, code_channel=code_channel)
            elif 'strict' in self.opt:
                strict = True if 'strict' not in self.opt else self.opt['strict']
                self.load_network(load_path_G, self.netG, strict=strict)
            else:
                self.load_network(load_path_G, self.netG)

    # Load vanilla network into SRIM by concatenating random weights
    def load_partial_network_concat(self, load_path, network, code_channel=5, strict=False):
        if isinstance(network, nn.DataParallel):
            network = network.module
        pretrained_dict = torch.load(load_path)
        model_dict = network.state_dict()
        weight_shape = pretrained_dict['model.0.weight'].shape
        concate_weights = torch.normal(mean=torch.zeros((weight_shape[0], code_channel, weight_shape[2],
                                                         weight_shape[3])), std=0.03)
        combined_weight = torch.cat((pretrained_dict['model.0.weight'], concate_weights), dim=1)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict['model.0.weight'] = combined_weight
        model_dict.update(pretrained_dict)
        network.load_state_dict(pretrained_dict, strict=strict)

    def load_partial_network(self, load_path, network, strict=False, exception=None):
        if isinstance(network, nn.DataParallel):
            network = network.module
        pretrained_dict = torch.load(load_path)
        model_dict = network.state_dict()
        weight_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and exception not in k}
        model_dict.update(weight_dict)
        network.load_state_dict(weight_dict, strict=strict)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
