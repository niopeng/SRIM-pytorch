import torch
import sys
import numpy as np
from dci import DCI


def print_without_newline(s):
   sys.stdout.write(s)
   sys.stdout.flush()


def find_min(codes):
    index = 0
    cur_min_index = 0
    cur_min = float('inf')
    for code in codes:
        if code < cur_min:
            cur_min = code
            cur_min_index = index
        index += 1
    return cur_min_index


def get_code(model, data, opt):
    options = opt['train']

    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])

    code_nc = int(opt['network_G']['in_code_nc'])
    pull_num_sample_per_img = int(options['num_code_per_img'])

    show_message = False if 'show_message' not in options else options['show_message']

    pull_gen_img = data['LR']
    real_gen_img = data['HR']
    pull_gen_code_0 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2], pull_gen_img.shape[3])

    if show_message:
        print("Generating Pull Samples")
    data_length = pull_gen_img.shape[0]

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0 and show_message:
            print_without_newline(
                '\rFinding first stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        if 'zero_code' in options and options['zero_code']:
            pull_gen_code_pool_0 = torch.zeros(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2],
                                               pull_gen_img.shape[3])
        elif 'rand_code' in options and options['rand_code']:
            pull_gen_code_pool_0 = torch.rand(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2],
                                               pull_gen_img.shape[3])
        else:
            pull_gen_code_pool_0 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2],
                                               pull_gen_img.shape[3])

        pull_img = pull_gen_img[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)

        cur_data = {'LR': pull_img, 'HR': real_gen_img[sample_index: sample_index + 1].expand(
            (pull_num_sample_per_img,) + real_gen_img.shape[1:])}

        model.feed_data(cur_data, code=pull_gen_code_pool_0)

        loss = model.get_loss()

        min_index = find_min(loss)

        pull_gen_code_0[sample_index] = pull_gen_code_pool_0[min_index]

    if show_message:
        print('\rFinding first stack code: Processed %d out of %d instances' % (
            data_length, data_length))

    if 'zero_code' in options and options['zero_code']:
        pull_gen_code_0 += sample_perturbation_magnitude * torch.zeros(pull_gen_img.shape[0], code_nc,
                                                                       pull_gen_img.shape[2],
                                                                       pull_gen_img.shape[3])
    elif 'rand_code' in options and options['rand_code']:
        pull_gen_code_0 += sample_perturbation_magnitude * torch.rand(pull_gen_img.shape[0], code_nc,
                                                                       pull_gen_img.shape[2],
                                                                       pull_gen_img.shape[3])
    else:
        pull_gen_code_0 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                       pull_gen_img.shape[2],
                                                                       pull_gen_img.shape[3])

    return pull_gen_code_0
