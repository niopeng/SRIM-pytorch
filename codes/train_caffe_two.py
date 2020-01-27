import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger, PrintLogger
from dci_util import *
import numpy as np


def validate(val_loader, opt, model, current_step, epoch, logger):
    print('---------- validation -------------')
    start_time = time.time()

    avg_psnr = 0.0
    idx = 0
    for val_data in val_loader:
        idx += 1
        img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
        img_dir = os.path.join(opt['path']['val_images'], img_name)
        util.mkdir(img_dir)

        if 'zero_code' in opt['train'] and opt['train']['zero_code']:
            code_val_0 = torch.zeros(val_data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                   val_data['LR'].shape[2] * 2, val_data['LR'].shape[3] * 2)
            code_val_1 = torch.zeros(val_data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                     val_data['LR'].shape[2] * 4, val_data['LR'].shape[3] * 4)
        elif 'rand_code' in opt['train'] and opt['train']['rand_code']:
            code_val_0 = torch.rand(val_data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                  val_data['LR'].shape[2] * 2, val_data['LR'].shape[3] * 2)
            code_val_1 = torch.rand(val_data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                    val_data['LR'].shape[2] * 4, val_data['LR'].shape[3] * 4)
        else:
            code_val_0 = torch.randn(val_data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                   val_data['LR'].shape[2] * 2, val_data['LR'].shape[3] * 2)
            code_val_1 = torch.randn(val_data['LR'].shape[0], int(opt['network_G']['in_code_nc']),
                                     val_data['LR'].shape[2] * 4, val_data['LR'].shape[3] * 4)
        model.feed_data(val_data, code=[code_val_0, code_val_1])
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'])  # uint8
        gt_img = util.tensor2img(visuals['HR'])  # uint8

        # Save SR images for reference
        save_img_path = os.path.join(img_dir, 'caffe_{:s}_x4_{:d}.png'.format(img_name, current_step))
        util.save_img(sr_img, save_img_path)

        # calculate PSNR
        crop_size = opt['scale']
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        avg_psnr += util.psnr(cropped_sr_img, cropped_gt_img)

    if current_step == 0:
        print('Saving the model at the end of iter {:d}.'.format(current_step))
        model.save(current_step)

    avg_psnr = avg_psnr / idx
    time_elapsed = time.time() - start_time
    # Save to log
    print_rlt = OrderedDict()
    print_rlt['model'] = opt['model']
    print_rlt['epoch'] = epoch
    print_rlt['iters'] = current_step
    print_rlt['time'] = time_elapsed
    print_rlt['psnr'] = avg_psnr
    logger.print_format_results('val', print_rlt)
    print('-----------------------------------')


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
        not key == 'pretrain_model_G' and not key == 'pretrain_model_D'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # print to file and std_out simultaneously
    sys.stdout = PrintLogger(opt['path']['log'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
            batch_size_per_month = dataset_opt['batch_size']
            batch_size_per_day = int(opt['datasets']['train']['batch_size_per_day'])
            num_month = int(opt['train']['num_month'])
            num_day = int(opt['train']['num_day'])

            use_dci = false if 'use_dci' not in opt['train'] else opt['train']['use_dci']
        elif phase == 'val':
            val_dataset_opt = dataset_opt
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    # create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    validate(val_loader, opt, model, current_step, 0, logger)
    for epoch in range(num_month):
        for i, train_data in enumerate(train_loader):
            cur_month_code = get_code_for_data_two(model, train_data, opt)
            for j in range(num_day):
                current_step += 1
                # get the sliced data
                cur_day_batch_start_idx = (j * batch_size_per_day) % batch_size_per_month
                cur_day_batch_end_idx = cur_day_batch_start_idx + batch_size_per_day
                if cur_day_batch_end_idx > batch_size_per_month:
                    cur_day_batch_idx = np.hstack((np.arange(cur_day_batch_start_idx, batch_size_per_month),
                                                   np.arange(cur_day_batch_end_idx - batch_size_per_month)))
                else:
                    cur_day_batch_idx = slice(cur_day_batch_start_idx, cur_day_batch_end_idx)

                cur_day_train_data = {'LR': train_data['LR'][cur_day_batch_idx],
                                      'HR': train_data['HR'][cur_day_batch_idx]}
                code = []
                for gen_code in cur_month_code:
                    code.append(gen_code[cur_day_batch_idx])

                # training
                model.feed_data(cur_day_train_data, code=code)
                model.optimize_parameters(current_step)

                time_elapsed = time.time() - start_time
                start_time = time.time()

                # log
                if current_step % opt['logger']['print_freq'] == 0:
                    logs = model.get_current_log()
                    print_rlt = OrderedDict()
                    print_rlt['model'] = opt['model']
                    print_rlt['epoch'] = epoch
                    print_rlt['iters'] = current_step
                    print_rlt['time'] = time_elapsed
                    for k, v in logs.items():
                        print_rlt[k] = v
                    print_rlt['lr'] = model.get_current_learning_rate()
                    logger.print_format_results('train', print_rlt)

                # save models
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    print('Saving the model at the end of iter {:d}.'.format(current_step))
                    model.save(current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    validate(val_loader, opt, model, current_step, epoch, logger)

                # update learning rate
                model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')


if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()
