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


def get_code_for_data(model, data, opt):
    options = opt['train']

    dci_num_comp_indices = int(options['dci_num_comp_indices'])
    dci_num_simp_indices = int(options['dci_num_simp_indices'])
    dci_num_levels = int(options['dci_num_levels'])
    dci_construction_field_of_view = int(options['dci_construction_field_of_view'])
    dci_query_field_of_view = int(options['dci_query_field_of_view'])
    dci_prop_to_visit = float(options['dci_prop_to_visit'])
    dci_prop_to_retrieve = float(options['dci_prop_to_retrieve'])
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

    out_feature_shape = model.netF(data['HR'][:1]).shape[1:]
    # initialize dci db
    pull_samples_dci_db = DCI(np.prod(out_feature_shape), dci_num_comp_indices, dci_num_simp_indices)

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

        # cur_data = {'LR': pull_img, 'HR': real_gen_img[sample_index: sample_index + 1]}
        cur_data = {'LR': pull_img, 'HR': real_gen_img[sample_index: sample_index + 1].expand(
            (pull_num_sample_per_img,) + real_gen_img.shape[1:])}

        model.feed_data(cur_data, code=pull_gen_code_pool_0)

        feature_output = model.get_features()

        pull_gen_features_pool = feature_output['gen_feat']
        target_feature = feature_output['gen_feat']

        pull_gen_features_pool = pull_gen_features_pool.reshape(-1, np.prod(
            pull_gen_features_pool.shape[1:])).double().numpy().copy()
        target_feature = target_feature.reshape(-1, np.prod(target_feature.shape[1:]))

        pull_samples_dci_db.add(pull_gen_features_pool,
                                num_levels=dci_num_levels,
                                field_of_view=dci_construction_field_of_view,
                                prop_to_visit=dci_prop_to_visit,
                                prop_to_retrieve=dci_prop_to_retrieve)
        pull_sample_idx_for_img, _ = pull_samples_dci_db.query(
            target_feature.numpy(),
            num_neighbours=1,
            field_of_view=dci_query_field_of_view,
            prop_to_visit=dci_prop_to_visit,
            prop_to_retrieve=dci_prop_to_retrieve)

        pull_gen_code_0[sample_index, :] = pull_gen_code_pool_0[int(pull_sample_idx_for_img[0][0]), :]
        # clear the db for next query
        pull_samples_dci_db.clear()

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


def get_code_for_data_two(model, data, opt):
    options = opt['train']

    dci_num_comp_indices = int(options['dci_num_comp_indices'])
    dci_num_simp_indices = int(options['dci_num_simp_indices'])
    dci_num_levels = int(options['dci_num_levels'])
    dci_construction_field_of_view = int(options['dci_construction_field_of_view'])
    dci_query_field_of_view = int(options['dci_query_field_of_view'])
    dci_prop_to_visit = float(options['dci_prop_to_visit'])
    dci_prop_to_retrieve = float(options['dci_prop_to_retrieve'])
    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])

    code_nc = int(opt['network_G']['in_code_nc'])
    pull_num_sample_per_img = int(options['num_code_per_img'])

    pull_gen_img = data['LR']
    d1_gen_img = data['D1']
    real_gen_img = data['HR']

    pull_gen_code_0 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 2, pull_gen_img.shape[3] * 2)

    forward_bs = 20

    print("Generating Pull Samples")
    data_length = pull_gen_img.shape[0]

    out_feature_shape = model.netF(data['D1'][:1])[-1].shape[1:]

    # ============ ADD POINT ==================
    pull_samples_dci_db = DCI(np.prod(out_feature_shape), dci_num_comp_indices, dci_num_simp_indices)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding first stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        pull_gen_code_pool_0 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 2, pull_gen_img.shape[3] * 2)
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)

        pull_gen_features_pool = []
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end]])
            feature_output = model.get_features()

            pull_gen_features_pool.append(feature_output['gen_feat_D1'].double().numpy())
        pull_gen_features_pool = np.concatenate(pull_gen_features_pool, axis=0)
        pull_gen_features_pool = pull_gen_features_pool.reshape(-1, np.prod(pull_gen_features_pool.shape[1:]))

        pull_samples_dci_db.add(pull_gen_features_pool.copy(),
                                num_levels=dci_num_levels,
                                field_of_view=dci_construction_field_of_view,
                                prop_to_visit=dci_prop_to_visit,
                                prop_to_retrieve=dci_prop_to_retrieve)
        target_feature = feature_output['real_feat_D1']
        target_feature = target_feature[0].reshape(1, np.prod(target_feature.shape[1:])).double().numpy().copy()

        pull_sample_idx_for_img, _ = pull_samples_dci_db.query(
            target_feature,
            num_neighbours=1,
            field_of_view=dci_query_field_of_view,
            prop_to_visit=dci_prop_to_visit,
            prop_to_retrieve=dci_prop_to_retrieve)

        pull_gen_code_0[sample_index, :] = pull_gen_code_pool_0[int(pull_sample_idx_for_img[0][0]), :]
        # clear the db
        pull_samples_dci_db.clear()

    print('\rFinding first stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    # ============ ADD POINT ==================
    pull_gen_code_1 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 4, pull_gen_img.shape[3] * 4)
    out_feature_shape = model.netF(data['HR'][:1])[-1].shape[1:]

    pull_samples_dci_db = DCI(np.prod(out_feature_shape), dci_num_comp_indices, dci_num_simp_indices)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding second stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        # ============ ADD POINT ==================
        pull_gen_code_pool_0 = pull_gen_code_0[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        pull_gen_features_pool = []
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end]])
            feature_output = model.get_features()

            pull_gen_features_pool.append(feature_output['gen_feat'].double().numpy())

        pull_gen_features_pool = np.concatenate(pull_gen_features_pool, axis=0)
        pull_gen_features_pool = pull_gen_features_pool.reshape(-1, np.prod(pull_gen_features_pool.shape[1:]))

        pull_samples_dci_db.add(pull_gen_features_pool.copy(),
                                num_levels=dci_num_levels,
                                field_of_view=dci_construction_field_of_view,
                                prop_to_visit=dci_prop_to_visit,
                                prop_to_retrieve=dci_prop_to_retrieve)
        target_feature = feature_output['real_feat'].double().numpy().copy()
        target_feature = target_feature[0].reshape(1, np.prod(target_feature.shape[1:]))

        pull_sample_idx_for_img, _ = pull_samples_dci_db.query(
            target_feature,
            num_neighbours=1,
            field_of_view=dci_query_field_of_view,
            prop_to_visit=dci_prop_to_visit,
            prop_to_retrieve=dci_prop_to_retrieve)

        pull_gen_code_1[sample_index, :] = pull_gen_code_pool_1[int(pull_sample_idx_for_img[0][0]), :]
        # clear the db
        pull_samples_dci_db.clear()

    print('\rFinding second stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    pull_gen_code_0 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 2,
                                                                   pull_gen_img.shape[3] * 2)
    pull_gen_code_1 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 4,
                                                                   pull_gen_img.shape[3] * 4)

    return [pull_gen_code_0, pull_gen_code_1]


def get_code_for_data_three(model, data, opt):
    options = opt['train']

    dci_num_comp_indices = int(options['dci_num_comp_indices'])
    dci_num_simp_indices = int(options['dci_num_simp_indices'])
    dci_num_levels = int(options['dci_num_levels'])
    dci_construction_field_of_view = int(options['dci_construction_field_of_view'])
    dci_query_field_of_view = int(options['dci_query_field_of_view'])
    dci_prop_to_visit = float(options['dci_prop_to_visit'])
    dci_prop_to_retrieve = float(options['dci_prop_to_retrieve'])
    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])

    code_nc = int(opt['network_G']['in_code_nc'])
    pull_num_sample_per_img = int(options['num_code_per_img'])

    pull_gen_img = data['LR']
    d1_gen_img = data['D1']
    d2_gen_img = data['D2']
    real_gen_img = data['HR']

    pull_gen_code_0 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 2, pull_gen_img.shape[3] * 2)

    forward_bs = 10

    print("Generating Pull Samples")
    data_length = pull_gen_img.shape[0]

    out_feature_shape = model.netF(data['D1'][:1])[-1].shape[1:]

    # ============ ADD POINT ==================
    pull_samples_dci_db = DCI(np.prod(out_feature_shape), dci_num_comp_indices, dci_num_simp_indices)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding first stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        pull_gen_code_pool_0 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 2, pull_gen_img.shape[3] * 2)
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 8,
                                           pull_gen_img.shape[3] * 8)
        pull_gen_features_pool = []
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            feature_output = model.get_features()

            pull_gen_features_pool.append(feature_output['gen_feat_D1'].double().numpy())
        pull_gen_features_pool = np.concatenate(pull_gen_features_pool, axis=0)
        pull_gen_features_pool = pull_gen_features_pool.reshape(-1, np.prod(pull_gen_features_pool.shape[1:]))

        pull_samples_dci_db.add(pull_gen_features_pool.copy(),
                                num_levels=dci_num_levels,
                                field_of_view=dci_construction_field_of_view,
                                prop_to_visit=dci_prop_to_visit,
                                prop_to_retrieve=dci_prop_to_retrieve)
        target_feature = feature_output['real_feat_D1']
        target_feature = target_feature[0].reshape(1, np.prod(target_feature.shape[1:])).double().numpy().copy()

        pull_sample_idx_for_img, _ = pull_samples_dci_db.query(
            target_feature,
            num_neighbours=1,
            field_of_view=dci_query_field_of_view,
            prop_to_visit=dci_prop_to_visit,
            prop_to_retrieve=dci_prop_to_retrieve)

        pull_gen_code_0[sample_index, :] = pull_gen_code_pool_0[int(pull_sample_idx_for_img[0][0]), :]
        # clear the db
        pull_samples_dci_db.clear()

    print('\rFinding first stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    # ============ ADD POINT ==================
    pull_gen_code_1 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 4, pull_gen_img.shape[3] * 4)
    out_feature_shape = model.netF(data['D2'][:1])[-1].shape[1:]

    pull_samples_dci_db = DCI(np.prod(out_feature_shape), dci_num_comp_indices, dci_num_simp_indices)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding second stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        # ============ ADD POINT ==================
        pull_gen_code_pool_0 = pull_gen_code_0[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_1 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 4,
                                           pull_gen_img.shape[3] * 4)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 8,
                                           pull_gen_img.shape[3] * 8)
        pull_gen_features_pool = []
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            feature_output = model.get_features()

            pull_gen_features_pool.append(feature_output['gen_feat_D2'].double().numpy())

        pull_gen_features_pool = np.concatenate(pull_gen_features_pool, axis=0)
        pull_gen_features_pool = pull_gen_features_pool.reshape(-1, np.prod(pull_gen_features_pool.shape[1:]))

        pull_samples_dci_db.add(pull_gen_features_pool.copy(),
                                num_levels=dci_num_levels,
                                field_of_view=dci_construction_field_of_view,
                                prop_to_visit=dci_prop_to_visit,
                                prop_to_retrieve=dci_prop_to_retrieve)
        target_feature = feature_output['real_feat_D2'].double().numpy().copy()
        target_feature = target_feature[0].reshape(1, np.prod(target_feature.shape[1:]))

        pull_sample_idx_for_img, _ = pull_samples_dci_db.query(
            target_feature,
            num_neighbours=1,
            field_of_view=dci_query_field_of_view,
            prop_to_visit=dci_prop_to_visit,
            prop_to_retrieve=dci_prop_to_retrieve)

        pull_gen_code_1[sample_index, :] = pull_gen_code_pool_1[int(pull_sample_idx_for_img[0][0]), :]
        # clear the db
        pull_samples_dci_db.clear()

    print('\rFinding second stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    # ============ ADD POINT ==================
    pull_gen_code_2 = torch.empty(pull_gen_img.shape[0], code_nc, pull_gen_img.shape[2] * 8, pull_gen_img.shape[3] * 8)
    out_feature_shape = model.netF(data['HR'][:1])[-1].shape[1:]

    pull_samples_dci_db = DCI(np.prod(out_feature_shape), dci_num_comp_indices, dci_num_simp_indices)

    for sample_index in range(data_length):
        if (sample_index + 1) % 10 == 0:
            print_without_newline(
                '\rFinding third stack code: Processed %d out of %d instances' % (
                    sample_index + 1, data_length))
        # ============ ADD POINT ==================
        pull_gen_code_pool_0 = pull_gen_code_0[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_1 = pull_gen_code_1[sample_index].expand(pull_num_sample_per_img, -1, -1, -1)
        pull_gen_code_pool_2 = torch.randn(pull_num_sample_per_img, code_nc, pull_gen_img.shape[2] * 8,
                                           pull_gen_img.shape[3] * 8)
        pull_gen_features_pool = []
        for i in range(0, pull_num_sample_per_img, forward_bs):
            pull_img = pull_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_target = real_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d1 = d1_gen_img[sample_index].expand(forward_bs, -1, -1, -1)
            pull_d2 = d2_gen_img[sample_index].expand(forward_bs, -1, -1, -1)

            cur_data = {'LR': pull_img, 'HR': pull_target, 'D1': pull_d1, 'D2': pull_d2}
            start = i
            end = i + forward_bs

            model.feed_data(cur_data, code=[pull_gen_code_pool_0[start:end], pull_gen_code_pool_1[start:end],
                                            pull_gen_code_pool_2[start:end]])
            feature_output = model.get_features()

            pull_gen_features_pool.append(feature_output['gen_feat'].double().numpy())

        pull_gen_features_pool = np.concatenate(pull_gen_features_pool, axis=0)
        pull_gen_features_pool = pull_gen_features_pool.reshape(-1, np.prod(pull_gen_features_pool.shape[1:]))

        pull_samples_dci_db.add(pull_gen_features_pool.copy(),
                                num_levels=dci_num_levels,
                                field_of_view=dci_construction_field_of_view,
                                prop_to_visit=dci_prop_to_visit,
                                prop_to_retrieve=dci_prop_to_retrieve)
        target_feature = feature_output['real_feat'].double().numpy().copy()
        target_feature = target_feature[0].reshape(1, np.prod(target_feature.shape[1:]))

        pull_sample_idx_for_img, _ = pull_samples_dci_db.query(
            target_feature,
            num_neighbours=1,
            field_of_view=dci_query_field_of_view,
            prop_to_visit=dci_prop_to_visit,
            prop_to_retrieve=dci_prop_to_retrieve)

        pull_gen_code_2[sample_index, :] = pull_gen_code_pool_2[int(pull_sample_idx_for_img[0][0]), :]
        # clear the db
        pull_samples_dci_db.clear()

    print('\rFinding third stack code: Processed %d out of %d instances' % (
        data_length, data_length))

    pull_gen_code_0 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 2,
                                                                   pull_gen_img.shape[3] * 2)
    pull_gen_code_1 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 4,
                                                                   pull_gen_img.shape[3] * 4)
    pull_gen_code_2 += sample_perturbation_magnitude * torch.randn(pull_gen_img.shape[0], code_nc,
                                                                   pull_gen_img.shape[2] * 8,
                                                                   pull_gen_img.shape[3] * 8)

    return [pull_gen_code_0, pull_gen_code_1, pull_gen_code_2]
