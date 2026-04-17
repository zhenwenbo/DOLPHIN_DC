import torch
import numpy as np
root_dir = '/DOLPHIN'
utils_dir = '/DOLPHIN'
import sys
sys.path.append(utils_dir)
from utils import loadBin
import time
import json
from collections import OrderedDict
import multiprocessing

data = "LASTFM"
layer=1
def calculate_staleness(eids, datas):
    srcs = datas['src'][eids].tolist()
    dsts = datas['dst'][eids].tolist()
    nodes = set(srcs) | set(dsts)
    return 2 * len(eids) - len(nodes) 

def calculate_delta(eid, node_set, datas):
    src = datas['src'][eid].item()
    dst = datas['dst'][eid].item()
    src_new = src not in node_set
    dst_new = dst not in node_set
    return 2 - sum([src_new, dst_new]) 

def check_duplicate_eids(eid_result):
    all_eids = []
    duplicate_info = {}
    total_num = 0
    for batch_idx, batch in enumerate(eid_result):
        for idx, eid in enumerate(batch):
            all_eids.append((eid, batch_idx, idx))
            total_num += len(batch)
    seen = {}
    has_duplicate = False
    for eid, batch_idx, idx in all_eids:
        if eid in seen:
            has_duplicate = True
            if eid not in duplicate_info:
                first_batch, first_idx = seen[eid]
                duplicate_info[eid] = [(first_batch, first_idx)]
            duplicate_info[eid].append((batch_idx, idx))
        else:
            seen[eid] = (batch_idx, idx)
    return has_duplicate, duplicate_info, total_num

def adaptive_split_in_large_batch(threshold, large_batch_data, cur_eid, datas):
    start = time.time()
    eid_result = []
    max_len_in_batch = 0
    while len(cur_eid) :
        nodes = set()
        cur_batch_eid = []
        initial_count = min(2000, len(cur_eid))
        initial_eids = cur_eid[:initial_count]
        
        for eid in initial_eids:
            src, dst = datas['src'][eid].item(), datas['dst'][eid].item()
            nodes.add(src)
            nodes.add(dst)
        del cur_eid[0:initial_count]
        cur_batch_eid = initial_eids
        batch_staleness = 2* initial_count - len(nodes) 
        eid_idx = 0
        incre_eid = []
        incre_eid_idx = []
        
        while eid_idx < len(cur_eid) and batch_staleness < threshold:
            eid = cur_eid[eid_idx]
            staleness = calculate_delta(eid, nodes, datas)
            if staleness == 2:
                eid_idx += 1
                continue
            else:
                initial_count += 1
                nodes.add(datas['src'][eid].item())
                nodes.add(datas['dst'][eid].item())
                batch_staleness += staleness 
                incre_eid.append(eid)
                incre_eid_idx.append(eid_idx)
                eid_idx += 1
        
        for eid in incre_eid:
            cur_batch_eid.append(eid)
        i = len(incre_eid_idx) - 1
        while i > -1:
            del cur_eid[incre_eid_idx[i]]
            i -= 1
        
        eid_idx = 0    
        while batch_staleness < threshold and eid_idx < len(cur_eid):
            initial_count += 1
            eid = cur_eid[eid_idx]
            batch_staleness += 2 
            cur_batch_eid.append(eid)
            eid_idx += 1
        del cur_eid[:eid_idx]
        if(initial_count > max_len_in_batch):
            max_len_in_batch = initial_count
        eid_result.append(cur_batch_eid)
   
    cur_eid = eid_result[-1]
    
    return eid_result


def calculate_single_small_batch(args):

    datas, small_batch_start, small_batch_end = args
    eids = datas['eid'][small_batch_start:small_batch_end].to(dtype=torch.int64)
    return calculate_staleness(eids, datas)


def process_single_batch(args):
    current_pos, large_batch_end, threshold, datas= args
    large_batch_length = large_batch_end - current_pos
    
    src_data = datas['src'][current_pos:large_batch_end]
    dst_data = datas['dst'][current_pos:large_batch_end]
    eid_data = datas['eid'][current_pos:large_batch_end]
    large_batch_edges = [(src.item(), dst.item()) for src, dst in zip(src_data, dst_data)]
    current_eids = [eid.item() for eid in eid_data]
    
    return adaptive_split_in_large_batch(threshold, large_batch_edges, current_eids, datas)

def process_split(current_pos, end_pos, large_batch_size, max_threshold, datas):
    batches_args = []
    temp_pos = current_pos
    first = True
    
    while temp_pos < end_pos:
        if first:
            first = False
            large_batch_end = min(((current_pos // large_batch_size) + 1) * large_batch_size,
                                 temp_pos + large_batch_size, end_pos)
            batches_args.append((temp_pos, large_batch_end, max_threshold, datas))
            temp_pos = large_batch_end
        else:
            large_batch_end = min(temp_pos + large_batch_size, end_pos)
            batches_args.append((temp_pos, large_batch_end, max_threshold, datas))
            temp_pos = large_batch_end
    
    num_workers = min(multiprocessing.cpu_count() - 1, len(batches_args)) 
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_single_batch, batches_args)
    
    batches = []
    for result in results:
        batches.extend(result)
    return batches, end_pos

def write_batches_to_bin(batches, bin_path):
    with open(bin_path, 'wb') as f:
        for batch in batches:
            batch_len = len(batch)
            np.array(batch_len, dtype=np.int32).tofile(f)
            if batch_len > 0:
                np.array(batch, dtype=np.int32).tofile(f)

def read_batches_from_bin(bin_path):
    batches = []
    with open(bin_path, 'rb') as f:
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            batch_len = np.frombuffer(len_bytes, dtype=np.int32)[0]
            batch = np.frombuffer(f.read(batch_len * 4), dtype=np.int32).tolist() if batch_len > 0 else []
            batches.append(batch)
    return batches

def process(datas, json_path):
    with open(json_path, 'r') as f:
        df_conf = json.load(f)
    
    total_size = len(datas['dst'])
    train_edge_end = df_conf['train_edge_end']
    val_edge_end = df_conf['val_edge_end']
    large_batch_size = 60000
    batch_size = 2000

    small_batch_tasks = []
    for large_batch_start in range(0, train_edge_end, large_batch_size):
        large_batch_end = min(large_batch_start + large_batch_size, train_edge_end)
        for small_batch_start in range(large_batch_start, large_batch_end, batch_size):
            small_batch_end = min(small_batch_start + batch_size, large_batch_end)
            small_batch_tasks.append((datas, small_batch_start, small_batch_end))
    
    num_workers = min(multiprocessing.cpu_count() - 1, len(small_batch_tasks))
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        all_thresholds = pool.map(calculate_single_small_batch, small_batch_tasks)
    
    
    max_threshold = max(all_thresholds) if all_thresholds else 0
    
    all_batches_train, current_pos = process_split(0, train_edge_end, large_batch_size, max_threshold, datas)
    
    return all_batches_train
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    
   
    base_path = f'/DOLPHIN/data/{data}'
    datas = {
        'src': loadBin(f'{base_path}/df-src.bin'),
        'dst': loadBin(f'{base_path}/df-dst.bin'),
        'eid': loadBin(f'{base_path}/df-eid.bin'),
    }
    json_path = f'{base_path}/df-conf.json'
    
    all_batches_train = process(datas, json_path)
    
    base_path = f'/DOLPHIN/eid/{data}'
    train_bin_path = f'{base_path}/{layer}/all_batches_train_final.bin'
    
    start = time.time()
    write_batches_to_bin(all_batches_train, train_bin_path)
    
    start = time.time()
    train_read = read_batches_from_bin(train_bin_path)
    
    eid_lengths = [len(eid) for eid in train_read]

    max_length = max(eid_lengths) if eid_lengths else 0
    above_threshold = sum(1 for l in eid_lengths if l <1000)
    total = len(eid_lengths)
    ratio_above = above_threshold / total if total > 0 else 0

    print(f"ratio: {ratio_above:.4f}")
    print(f"{total} batches")