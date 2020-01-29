import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss


class SRIMMultCodeModel(BaseModel):
    def __init__(self, opt):
        super(SRIMMultCodeModel, self).__init__(opt)
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
                # selected 'conv2_2' and 'conv4_4' in the VGG19 activation here, change by 'feat_layers'
                self.netF = networks.define_F(opt, use_bn=False, feat_layers=[7, 25]).to(self.device)
                self.feat_weights = [1, 0.1]

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

    def feed_data(self, data, need_HR=True, code=[]):
        # LR
        self.var_L = data['LR'].to(self.device)

        self.code = code[0].to(self.device) if len(code) > 0 else None
        self.code_1 = code[1].to(self.device) if len(code) > 1 else None
        self.code_2 = code[2].to(self.device) if len(code) > 2 else None

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
            self.var_D1 = data['D1'].to(self.device) if 'D1' in data else None
            self.var_D2 = data['D2'].to(self.device) if 'D2' in data else None

    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.code, self.code_1, self.code_2)[-1]
        l_g_total = 0

        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
            l_g_total += l_g_pix
        if self.cri_fea:  # feature loss
            real_fea = self.netF(self.var_H)
            fake_fea = self.netF(self.fake_H)
            if type(real_fea) is list:
                l_g_fea = 0
                for i in range(len(real_fea)):
                    l_g_fea += self.cri_fea(fake_fea[i], real_fea[i].detach()) * self.feat_weights[i]
            else:
                l_g_fea = self.cri_fea(fake_fea, real_fea.detach())
            l_g_total += self.l_fea_w * l_g_fea

        l_g_total.backward()
        self.optimizer_G.step()

        # G
        if self.cri_pix:
            self.log_dict['l_g_pix'] = l_g_pix.item()
        if self.cri_fea:
            self.log_dict['l_g_fea'] = l_g_fea.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, self.code, self.code_1, self.code_2)[-1]
        self.netG.train()

    def get_features(self):
        self.netG.eval()
        out_dict = OrderedDict()
        fake_D1 = None
        fake_D2 = None
        with torch.no_grad():
            outputs = self.netG(self.var_L, self.code, self.code_1, self.code_2)
            self.fake_H = outputs[-1]
            fake_D1 = outputs[0] if len(outputs) > 1 else None
            fake_D2 = outputs[1] if len(outputs) > 2 else None
        gen_feat = self.netF(self.fake_H)
        real_feat = self.netF(self.var_H)
        out_dict['gen_feat'] = gen_feat[-1].data.double().cpu() if type(
            gen_feat) is list else gen_feat.detach().double().cpu()
        out_dict['real_feat'] = real_feat[-1].data.double().cpu() if type(
            gen_feat) is list else real_feat.detach().double().cpu()
        if fake_D1 is not None:
            gen_feat_D1 = self.netF(fake_D1)
            real_feat_D1 = self.netF(self.var_D1)
            out_dict['gen_feat_D1'] = gen_feat_D1[-1].data.double().cpu() if type(
                gen_feat_D1) is list else gen_feat_D1.detach().double().cpu()
            out_dict['real_feat_D1'] = real_feat_D1[-1].data.double().cpu() if type(
                gen_feat_D1) is list else real_feat_D1.detach().double().cpu()
        if fake_D2 is not None:
            gen_feat_D2 = self.netF(fake_D2)
            real_feat_D2 = self.netF(self.var_D2)
            out_dict['gen_feat_D2'] = gen_feat_D2[-1].data.double().cpu() if type(
                gen_feat_D2) is list else gen_feat_D2.detach().double().cpu()
            out_dict['real_feat_D2'] = real_feat_D2[-1].data.double().cpu() if type(
                gen_feat_D2) is list else real_feat_D2.detach().double().cpu()
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

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            if 'load_partial' in self.opt and self.opt['load_partial']:
                self.load_partial_network(load_path_G, self.netG)
            else:
                self.load_network(load_path_G, self.netG)

    def load_partial_network(self, load_path, network, strict=False):
        if isinstance(network, nn.DataParallel):
            network = network.module
        pretrained_dict = torch.load(load_path)
        model_dict = network.state_dict()
        for k, v in model_dict.items():
            if k in pretrained_dict:
                model_dict.update({k: pretrained_dict[k]})
        network.load_state_dict(pretrained_dict, strict=strict)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
