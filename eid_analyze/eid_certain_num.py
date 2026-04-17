import torch
import numpy as np
utils_dir = '/DOLPHIN'
import sys
sys.path.append(utils_dir)
from utils import loadBin
import time
import json


def process(d):
    base_path = f'/DOLPHIN/data/{d}'
    datas = {
        'src': loadBin(f'{base_path}/df-src.bin'),
        'dst': loadBin(f'{base_path}/df-dst.bin'),
        'eid': loadBin(f'{base_path}/df-eid.bin'), 
    }
    json_path = f'/DOLPHIN/data/{d}/df-conf.json'

    with open(json_path, 'r') as f:
        df_conf = json.load(f)

    train_edge_end = df_conf['train_edge_end']
    val_edge_end = df_conf['val_edge_end']
    total_size = len(datas['eid'])

    fixed_batch_size = 2000  
    first_val_block_flag = True
    first_test_block_flag = True
    first_val_block = ((train_edge_end // 60000) + 1) * 60000 
    first_test_block = ((val_edge_end // 60000) + 1) * 60000
    all_batches_train = []
    all_batches_val = []
    all_batches_test = []


    current_pos = 0
    while current_pos < train_edge_end:
        batch_end = min(current_pos + fixed_batch_size, train_edge_end)
        batch_length = batch_end - current_pos
        batch_eids = datas['eid'][current_pos:batch_end]
        batch = [eid.item() for eid in batch_eids]
        all_batches_train.append(batch)

        if current_pos % (fixed_batch_size * 10) == 0:
            progress = (current_pos / train_edge_end) * 100 if train_edge_end !=0 else 0
            print(f"ratio: {progress:.2f}%")
        current_pos = batch_end

   
   
    while current_pos < val_edge_end:
        if(batch_end == first_val_block):
            first_val_block_flag = False
        if(first_val_block_flag):
            batch_end = min(current_pos + fixed_batch_size, first_val_block)
        else:
            batch_end = min(current_pos + fixed_batch_size, val_edge_end)
        batch_length = batch_end - current_pos
        
        batch_eids = datas['eid'][current_pos:batch_end]
        batch = [eid.item() for eid in batch_eids]
        all_batches_val.append(batch)
        
        if current_pos % (fixed_batch_size * 10) == 0:
            progress = ((current_pos - train_edge_end) / (val_edge_end - train_edge_end)) * 100 if (val_edge_end - train_edge_end)!=0 else 0
            print(f"ratio: {progress:.2f}%")
        current_pos = batch_end

    
    while current_pos < total_size:
        if(batch_end == first_test_block):
            first_test_block_flag = False
        if(first_test_block_flag):
            batch_end = min(current_pos + fixed_batch_size, first_test_block)
        else:
            batch_end = min(current_pos + fixed_batch_size, total_size)
        batch_length = batch_end - current_pos
        batch_eids = datas['eid'][current_pos:batch_end]
        batch = [eid.item() for eid in batch_eids]
        all_batches_test.append(batch)
        if current_pos % (fixed_batch_size * 10) == 0:
            progress = ((current_pos - val_edge_end) / (total_size - val_edge_end)) * 100 if (total_size - val_edge_end)!=0 else 0
            print(f"ratio : {progress:.2f}%")
        current_pos = batch_end

    return all_batches_train, all_batches_val, all_batches_test



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
            if batch_len > 0:
                data_bytes = f.read(batch_len * 4) 
                batch = np.frombuffer(data_bytes, dtype=np.int32).tolist()
            else:
                batch = []
            batches.append(batch)
    return batches      


if __name__ == "__main__":
    data = 'LASTFM'
    layer = 1
    time_process_start = time.time()
    train_eid_list, val_eid_list, test_eid_list = process(data)
    time_process = time.time() - time_process_start
    print(f"\ntotal time:{time_process:.2f}s")


    base_path = f'/DOLPHIN/eid/{data}'
    train_bin_path = f'{base_path}/1/all_batches_train_2000.bin'
    val_bin_path = f'{base_path}/1/all_batches_val_2000.bin'
    test_bin_path = f'{base_path}/1/all_batches_test_2000.bin'
    
    start = time.time()
    write_batches_to_bin(train_eid_list, train_bin_path)
    write_batches_to_bin(val_eid_list, val_bin_path)
    write_batches_to_bin(test_eid_list, test_bin_path)
    write_time = time.time() - start
    print(f"write in:{write_time:.2f}s")

