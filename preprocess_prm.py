import os
import pickle
import pickle as pkl
from datetime import date
import random
from collections import defaultdict
import numpy as np


# map old feature id to new one
def convert(data, item_ft_map, uid_map):
    uid, profile, spar_ft, dens_ft1, dens_ft2, label = [eval(v) for v in data.strip().split('|')]
    dens_ft = np.concatenate((np.array(dens_ft1), np.array(dens_ft2)), axis=1).tolist()
    for i in range(len(spar_ft)):
        spar_ft[i] = [item_ft_map[j][spar_ft[i][j]] for j in range(len(item_ft_map))]
    return uid_map[uid], spar_ft, dens_ft, label


def get_data_with_hist(idx_list, max_hist_len, num, records, uid_map, item_ft_map):
    keep_uid, keep_lb, keep_dens, keep_spar, hist_spar, hist_dens = [], [], [], [], [], []
    idx_list_len = len(idx_list)
    for i in range(idx_list_len - num, idx_list_len):
        uid, spar_ft, dens_ft, label = convert(records[idx_list[i]], item_ft_map, uid_map)
        keep_uid.append(uid)
        keep_spar.append(spar_ft)
        keep_dens.append(dens_ft)
        keep_lb.append(label)
    begin, end = max(0, len(idx_list) - max_hist_len - num), len(idx_list) - 1
    for i in range(begin, end):
        uid, spar_ft, dens_ft, label = convert(records[idx_list[i]], item_ft_map, uid_map)
        for j in range(len(label)):
            if label[j]:
                hist_spar.append(spar_ft[j])
                hist_dens.append(dens_ft[j])
    hist_dens.reverse()
    hist_spar.reverse()
    return keep_uid, keep_spar, keep_dens, hist_spar[:max_hist_len+num-1:], hist_dens[:max_hist_len+num-1:], keep_lb




def process_data(raw_dir1, raw_dir2, store_dir):
    fin1 = open(raw_dir1, 'r')
    records = fin1.readlines()
    fin2 = open(raw_dir2, 'r')
    records.extend(fin2.readlines())
    print('finish loading data')

    uid_idx = defaultdict(list)
    with_null = 0

    # remove records with null feature
    for i, v in enumerate(records):
        if v.find('null') != -1:
            with_null += 1
            continue
        uid_idx[eval(v.strip().split('|')[0])].append(i)

    print('origin\nuser num:', len(uid_idx), 'record num:', len(records), 'with null:', with_null)

    # counting the number of feature
    uid, profile, spar_ft, dens_ft1, dens_ft2, label = records[0].strip().split('|')
    profile_fnum, itm_dens_fnum, itm_spar_fnum = len(eval(profile)), len(eval(dens_ft1)[0]) + len(eval(dens_ft2)[0]), len(eval(spar_ft)[0])
    print('profile fnum:', profile_fnum, 'itm sparse fnum:', itm_spar_fnum, 'itm dense fnum:', itm_dens_fnum)

    remain_idx = []
    idx_uid = {}
    uid_set, item_ft_set, usr_ft_set = set(), [set() for _ in range(itm_spar_fnum)], [set() for _ in range(profile_fnum)]
    # num4, num5, num6, num7, num8 = 0, 0, 0, 0, 0
    pos, neg = 0, 0
    seq_min, seq_max, total_len = 1e9, 0, 0


    # filter
    for k, v in uid_idx.items():
        # if len(v) > 8:
        #     num8 += 1
        # elif len(v) > 7:
        #     num7 += 1
        # elif len(v) > 6:
        #     num6 += 1
        # elif len(v) > 5:
        #     num5 += 1
        # elif len(v) > 4:
        #     num4 += 1
        if len(v) > 3:
            uid_set.add(k)
            for idx in v:
                remain_idx.append(idx)
                idx_uid[idx] = k

                uid, profile, spar_ft, dens_ft1, dens_ft2, label = records[idx].strip().split('|')
                label = eval(label)
                seq_min, seq_max = min(seq_min, len(label)), max(seq_max, len(label))
                # total_len += len(label)
                for i, value in enumerate(eval(profile)):
                    usr_ft_set[i].add(value)
                for ft in eval(spar_ft):
                    for i, value in enumerate(ft):
                        item_ft_set[i].add(value)

                pos += sum(label)
                neg += len(label) - sum(label)

    # print(' >100:', num100, '\n50-100:', num50, '\n20-50', num20, '\n10-20:', num10)
    # print(' >8:', num8, '\n7:', num7, '\n6', num6, '\n5:', num5, '\n4:', num4)
    print('after filter < 4')
    print('user num:', len(uid_set), 'list num:', len(idx_uid))
    print('user ft:', len(usr_ft_set[0]), len(usr_ft_set[1]), len(usr_ft_set[2]))
    # print(usr_ft_set)
    print('item ft', len(item_ft_set[0]), len(item_ft_set[1]), len(item_ft_set[2]), len(item_ft_set[3]), len(item_ft_set[4]))
    # print(item_ft_set[1:])
    print('pos/neg:', pos*1.0/neg)
    # print('max seq len:', seq_max, 'min seq len:', seq_min, 'average len:', total_len/len(remain_idx))

    # rename feature id
    ft_id = 1
    uid_map, usr_ft_map, item_ft_map = {}, [{} for _ in range(profile_fnum)], [{} for _ in range(itm_spar_fnum)]

    for v in uid_set:
        uid_map[v] = ft_id
        ft_id += 1
    for i, ft in enumerate(usr_ft_set):
        for v in ft:
            usr_ft_map[i][v] = ft_id
            ft_id += 1
    for i, ft in enumerate(item_ft_set):
        for v in ft:
            item_ft_map[i][v] = ft_id
            ft_id += 1

    # print('user_ft_map', usr_ft_map)
    # print('itm_ft_map', item_ft_map[2:])

    stat = {'user_num': len(uid_map), 'item_num': len(item_ft_map[0]), 'cate_num': len(item_ft_map[1]),
            'ft_num': ft_id, 'list_num': len(idx_uid), 'user_fnum': 1, 'profile_fnum': profile_fnum,
            'itm_spar_fnum': itm_spar_fnum, 'itm_dens_fnum': itm_dens_fnum}
    with open(store_dir + 'data.stat', 'wb') as f:
        pkl.dump(stat, f)

    # count clicks of different category at each position & process user profile
    prop = {}
    uid_profile = {}
    for idx in remain_idx:
        uid, profile, spar_ft, dens_ft1, dens_ft2, label = records[idx].strip().split('|')
        uid, profile, spar_ft, label = eval(uid), eval(profile), eval(spar_ft), eval(label)
        for i, sft in enumerate(spar_ft):
            cate = item_ft_map[1][sft[1]]
            if not prop.__contains__(cate):
                prop[cate] = [0 for i in range(seq_max)]
            prop[cate][i] += label[i]
        if not uid_profile.__contains__(uid_map[uid]):
            uid_profile[uid_map[uid]] = [usr_ft_map[i][profile[i]] for i in range(profile_fnum)]

    for k, v in prop.items():
        first_pos_click = v[0]
        for i in range(seq_max):
            prop[k][i] = (v[i] + 1e-6) / (first_pos_click + 1e-6)

    with open(store_dir + 'prop', 'wb') as f:
        pkl.dump(prop, f)

    with open(store_dir + 'user.profile', 'wb') as f:
        pkl.dump(uid_profile, f)

    prop, uid_profile = [], []
    print('save prop & user profile')

    # split data
    train_list, val_list, test_list = [], [], []
    for uid in uid_set:
        idx_list = uid_idx[uid]
        rand = random.random()
        uid, ft_spar, ft_dens, hist_spar, hist_dens, label = get_data_with_hist(idx_list, max_hist_len, 1, records, uid_map, item_ft_map)
        if rand < 0.1:
            test_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar, hist_dens, label[0]])
        elif rand < 0.6:
            val_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar, hist_dens, label[0]])
        else:
            train_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar, hist_dens, label[0]])
        # if random.random() < 0.1:
        #     uid, ft_spar, ft_dens, hist_spar, hist_dens, label = get_data_with_hist(idx_list, max_hist_len, 3, records, uid_map, item_ft_map)
        #     train_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar[-max_hist_len-2:-2], hist_dens[-max_hist_len-2:-2], label[0]])
        #     val_list.append([uid[1], ft_spar[1], ft_dens[1], hist_spar[-max_hist_len-1:-1], hist_dens[-max_hist_len-1:-1], label[1]])
        #     test_list.append([uid[2], ft_spar[2], ft_dens[2], hist_spar[-max_hist_len:], hist_dens[-max_hist_len:], label[2]])
        # # elif len(idx_list) == 2:
        # #     uid, ft_spar, ft_dens, hist_spar, hist_dens, label = get_data_with_hist(idx_list, max_hist_len, 1, records, uid_map, item_ft_map)
        # #     if random.random() < 0.1:
        # #         test_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar, hist_dens, label[0]])
        # #     else:
        # #         val_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar, hist_dens, label[0]])
        # else:
        #     uid, ft_spar, ft_dens, hist_spar, hist_dens, label = get_data_with_hist(idx_list, max_hist_len, 2, records, uid_map, item_ft_map)
        #     if random.random() < 0.1:
        #         val_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar[-max_hist_len-1:-1], hist_dens[-max_hist_len-1:-1], label[0]])
        #         test_list.append([uid[1], ft_spar[1], ft_dens[1], hist_spar[-max_hist_len:], hist_dens[-max_hist_len:], label[1]])
        #     else:
        #         train_list.append([uid[0], ft_spar[0], ft_dens[0], hist_spar[-max_hist_len-1:-1], hist_dens[-max_hist_len-1:-1], label[0]])
        #         val_list.append([uid[1], ft_spar[1], ft_dens[1], hist_spar[-max_hist_len:], hist_dens[-max_hist_len:], label[1]])

    print('train num', len(train_list), 'val num', len(val_list), 'test num', len(test_list))
    with open(store_dir + 'data.data', 'wb') as f:
        pkl.dump([train_list, val_list, test_list], f)


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    data_dir = 'data/'
    data_set_name = 'prm'
    max_hist_len = 30
    list_len = 30
    raw_dir1 = os.path.join(data_dir, data_set_name + '/raw_data/set1.train.txt.part1')
    raw_dir2 = os.path.join(data_dir, data_set_name + '/raw_data/set1.train.txt.part2')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed5/')

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if not os.path.isfile(processed_dir + 'data.data'):
        process_data(raw_dir1, raw_dir2, processed_dir)
