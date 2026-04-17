
import torch
import dgl
import numpy as np
import pandas as pd
import time
import sys
import os
total_start = time.time()
root_dir = '/DOLPHIN'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config.train_conf import *
GlobalConfig.conf = 'basic_conf.json'

from utils import *
from sampler import *

from sampler.sampler_gpu import *


def get_max_edge_num(indptr):
    dif = torch.diff(indptr)
    ind = torch.nonzero(dif > 10).reshape(-1)
    dif[ind] = 10

    return torch.sum(dif)

def gen_expire(args):

    d = args.data
    zombie_block = args.zombie_block
    batch_size = args.bs

    g, datas, df_conf = load_graph_bin(args.data)


    fan_nums = [10]
    layers = len(fan_nums)
    sampler_gpu = Sampler_GPU(g, fan_nums, layers)


    expired = torch.zeros_like(sampler_gpu.indices).cuda()

    cal_max_edge_num = 0
    cal_edge_num = 0
    max_edge_num = get_max_edge_num(sampler_gpu.indptr) + batch_size * 3

    totaleid = sampler_gpu.totaleid.cuda()

    eid_flag = torch.zeros_like(totaleid)
    eid_uni, counts = torch.unique(totaleid, return_counts = True)
    eid_uni, indices = torch.sort(eid_uni)
    counts = counts[indices]
    eid_flag = counts

    start_eid = 0
    end_eid = 0
    end_ptr = 0

    map = torch.zeros(max_edge_num, dtype = torch.int32, device = 'cuda:0') + (2**31 - 1) 

    exp_eids = None

    # node_feats, edge_feats = load_feat(d)
    path = f'/DOLPHIN/data/{d}'

    part_path = path + f'/part-{batch_size}'

    if not os.path.exists(part_path):
        os.mkdir(part_path)

    unexpire_ind = torch.empty(0, dtype = torch.int32, device = 'cuda:0')


    left, right = 0, 0
    batch_num = 0
    total_src = datas['src'].cuda()
    total_dsts = datas['dst'].cuda()
    total_time = datas['time'].to(torch.float32).cuda()
    edge_end = datas['src'].shape[0]
    while True:
        total_s = time.time()
        right += batch_size
        right = min(edge_end, right)
        if (left >= right):
            break

        src = total_src[left: right]
        dst = total_dsts[left: right]
        times = total_time[left: right]

        root_nodes = torch.cat([src, dst])
        root_ts = torch.cat([times, times]).to(torch.float32)

        start_eid = end_eid
        end_eid += root_nodes.shape[0] // 2

        cur_eids = torch.arange(start_eid, end_eid, dtype = torch.int32, device='cuda:0')
       
        ind_se = torch.tensor([start_eid, end_eid], dtype = torch.int32)
        saveBin_concurrent(ind_se, part_path + f'/part{batch_num}_edge_incre_bound.pt' )

        judge_s = time.time()
        replace_idx = None
        if (exp_eids == None):
            map[:cur_eids.shape[0]] = cur_eids
            end_ptr = end_ptr + cur_eids.shape[0]
            replace_idx = torch.arange(cur_eids.shape[0], dtype = torch.int32)
        else:
            exp_eids_sort,_ = torch.sort(exp_eids)
            map_sort,map_sort_indices = torch.sort(map)
            table1 = torch.zeros_like(exp_eids_sort) - 1
            table2 = torch.zeros_like(map_sort) - 1
            dgl.findSameIndex(exp_eids_sort, map_sort, table1, table2)
            table1 = map_sort_indices[table1.long()]

            unalloc_ptr = None
            if (torch.nonzero(table1 == -1).reshape(-1).shape[0] > 0):
                print(f"error!")
        
            else:
                replace_idx = torch.zeros_like(cur_eids)
        
                if (table1.shape[0] >= cur_eids.shape[0]):
                    cur_unexpire = table1[cur_eids.shape[0]:].to(torch.int32)
                    unexpire_ind = torch.cat((unexpire_ind, cur_unexpire))
                    replace_idx[:cur_eids.shape[0]] = table1[:cur_eids.shape[0]]
                elif (table1.shape[0] + unexpire_ind.shape[0] >= cur_eids.shape[0]):
                    use_expire_num = cur_eids.shape[0] - table1.shape[0]
                    use_expire = unexpire_ind[:use_expire_num]
                    unexpire_ind = unexpire_ind[use_expire_num:]
                    replace_idx[:table1.shape[0]] = table1
                    replace_idx[table1.shape[0]:] = use_expire
                else:    
                    unalloc_ptr = table1.shape[0] + unexpire_ind.shape[0] 
                    replace_idx[:table1.shape[0]] = table1
                    if (unexpire_ind.shape[0] > 0):
                        replace_idx[table1.shape[0]:table1.shape[0] + unexpire_ind.shape[0]] = unexpire_ind
                        unexpire_ind = torch.empty(0, dtype = torch.int32, device='cuda:0')

                    replace_idx[unalloc_ptr:] = torch.arange(end_ptr, end_ptr + (replace_idx.shape[0] - unalloc_ptr), dtype = torch.int32, device = 'cuda:0')
                
                
                replace_idx = replace_idx.to(torch.int64)

                if (unalloc_ptr is None):
                    unalloc_ptr = cur_eids.shape[0]
                end_ptr = end_ptr + (replace_idx.shape[0] - unalloc_ptr)
                if (torch.max(replace_idx) >= map.shape[0]):
                    map = torch.cat((map, torch.zeros(torch.max(replace_idx) - map.shape[0] + 1, dtype = torch.int32, device = 'cuda:0') + (2**31 - 1) ))
                map[replace_idx] = cur_eids

        save_s = time.time()
        if (batch_num == 0):
            saveBin_concurrent(map.cpu(), part_path + f'/part{batch_num}_edge_incre_map.pt')
        if (replace_idx is not None):
            saveBin_concurrent(replace_idx.cpu(), part_path + f'/part{batch_num}_edge_incre_replace.pt')
        start = time.time()
        expired_clone = expired.clone()
        ret_list = sampler_gpu.sample_layer(root_nodes, root_ts, expired=expired, sample_mode = 'expire', sample_param={'cur_block': batch_num, 'zombie_block': zombie_block})
    
        expired_cur = expired ^ expired_clone

        ind = torch.nonzero(expired_cur).reshape(-1)
        exp_eids = totaleid[ind]

        exp_eids, counts = torch.unique(exp_eids, return_counts = True)
        eid_flag_clone = eid_flag.clone()
        eid_flag[exp_eids.long()] -= counts
        exp_eids = torch.nonzero( (eid_flag ^ eid_flag_clone).to(torch.bool) & (eid_flag == 0) ).to(torch.int32).reshape(-1)

        cal_edge_num += root_nodes.shape[0] // 2
        cal_max_edge_num = max(cal_edge_num, cal_max_edge_num)
       
        
        cal_edge_num -= exp_eids.shape[0]


        left = right
        batch_num += 1
    flush_saveBin_conf()
    return end_ptr

import argparse
import os
import json

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='LASTFM')
parser.add_argument('--bs', type=int, help='batch size', default='60000')
parser.add_argument('--zombie_block', type=int, help='zombie block', default='2')
args=parser.parse_args()

max_edge_num = gen_expire(args)
data = {args.data: max_edge_num}

file_path = f'/DOLPHIN/preprocessing/expire-{args.bs}.json'

if os.path.exists(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        existing_data = json.load(file)
    
    
    existing_data.update(data)
else:
    
    existing_data = data


with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(existing_data, file, ensure_ascii=False, indent=4)

print(f"memory_preprocess total use time{time.time() - total_start:.4f}s")