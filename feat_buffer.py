import dgl
import torch
import numpy as np
import time
from sampler.sampler_gpu import *
from config.train_conf import *
import os
from utils import *
import concurrent.futures
from queue import Queue
from multiprocessing import Queue
import multiprocessing
import json
import gc
import math


class Feat_buffer:
    def __init__(self, d, df, datas, train_param, memory_param, train_edge_end, presample_batch, block_size,sampler, neg_sampler, node_num = None, edge_num = None, prefetch_conn = None, feat_dim = None, substream_size = None):
        self.d = d
        self.df = df
        self.datas = datas
        self.sampler = sampler
        self.neg_sampler = neg_sampler
        self.presample_batch = presample_batch
        self.train_param = train_param
        self.block_size = block_size
        batch_size = train_param['batch_size'] * presample_batch
        train_batch_size = train_param['batch_size']
        self.path = f'/DOLPHIN/data/{d}'
        self.total_node_num = node_num
        self.total_edge_num = edge_num

        with open('./cur_train.json', mode='r') as f:
            conf = json.load(f)
        self.node_feat_type = conf['node_feat_type']
        self.edge_feat_type = conf['edge_feat_type']
        self.dataset = conf['dataset']
        self.node_num = conf['node_num']
        self.edge_num = conf['edge_num']

        self.node_feat_dim, self.edge_feat_dim = feat_dim
        self.has_ef = self.edge_feat_dim> 0
        self.memory_param = memory_param
        self.use_memory = memory_param['type'] != 'none'
        
        self.transfer_cpu = sampler.layer != 1 

        self.batch_size = batch_size 
        self.train_batch_size = train_batch_size
        self.train_edge_end = train_edge_end

        if (substream_size is not None):
            self.batch_size = substream_size

        assert batch_size % train_batch_size == 0 
        self.batch_num = batch_size / train_batch_size 

        self.batch_num = batch_size // train_batch_size
       

        self.cur_batch = 0 

        self.cur_mode = 'train'
        self.use_buffer = True

        self.part_edge_feats = None
        self.part_edge_map = None
        self.part_node_feats = None
        self.part_node_map = None

        self.part_memory = None
        self.part_memory_ts = None
        self.part_mailbox = None
        self.part_mailbox_ts = None

        self.part_memory_map = None
        self.mode = ''
        self.io_time = 0
        self.io_mem = 0
        
        self.time_load = 0
        self.time_analyze = 0
        self.time_presample = 0
        self.time_pre_neg_sample = 0
        self.time_exec_mem = 0
        self.time_neg_analyze = 0
        self.time_refresh = 0



        self.err_num = 0
        use_detection = False

        self.err_detection = use_detection
        if (self.err_detection):
            self.det_node_feats, self.det_edge_feats = load_feat(self.d)


        self.mem_detection = False
        if (self.mem_detection):
            self.det_memory = torch.zeros((node_num, memory_param['dim_out']), dtype=torch.float32)
            self.det_memory_ts = torch.zeros((node_num), dtype=torch.float32)
            self.det_mailbox = torch.zeros((node_num, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + self.edge_feat_dim), dtype=torch.float32)
            self.det_mailbox_ts = torch.zeros((node_num, memory_param['mailbox_size']), dtype=torch.float32)

            
        self.config = GlobalConfig()
        self.use_disk = self.config.use_disk
        self.data_incre = self.config.data_incre
        self.preFetchExecutor = concurrent.futures.ThreadPoolExecutor(2) 



        if (d == 'LASTFM'):
            self.share_edge_num = 500000
            self.share_node_num = 20000
            self.tmp_tensor_num = 20000000
        elif (d == 'TALK'):
            self.share_edge_num = 800000
            self.share_node_num = 200000
            self.tmp_tensor_num = 100000000
        else:
            self.share_edge_num = 3200000
            self.share_node_num = 2000000
            self.tmp_tensor_num = 500000000



        if (d == 'MAG' or d == 'MOOC'):
            self.share_edge_num = 0

        self.bucket_config = None
        self.bucket_config_path = f'{self.path}/part-{self.batch_size}-{self.sampler.fan_nums}/bucket_config.json'
        if (os.path.exists(self.bucket_config_path)):
            with open(self.bucket_config_path, 'r') as json_file:
                self.bucket_config = json.load(json_file)
            self.bucket_cache_node_num = self.bucket_config['bucket_cache_node_num'] if 'bucket_cache_node_num' in self.bucket_config else None
            self.bucket_cache_edge_num = self.bucket_config['bucket_cache_edge_num']if 'bucket_cache_edge_num' in self.bucket_config else None
            # self.share_edge_num = self.bucket_config['max_edge_num'] * 2
        #     self.share_node_num = self.bucket_config['max_node_num'] * 2
        #     self.tmp_tensor_num = 500000000

        #     self.bucket_ptr = self.bucket_config['bucket_ptr']
        
        if (prefetch_conn[0] is None):
            self.prefetch_conn = None
        else:
            self.prefetch_conn, self.prefetch_only_conn = prefetch_conn
            self.init_share_tensor()
        
        self.preFetchDataCache = Queue()
        self.cur_block = 0
        
        if (self.use_disk):
            self.edge_feats_path = f'/DOLPHIN/data/{self.d}/edge_features.bin'
            self.node_feats_path = f'/DOLPHIN/data/{self.d}/node_features.bin'

            if ('mailbox_size' in memory_param):
                self.memory_shape = [memory_param['dim_out']]
                self.memory_ts_shape = [1]
                self.mailbox_shape = [memory_param['mailbox_size'], 2 * memory_param['dim_out'] + self.edge_feat_dim]
                self.mailbox_ts_shape = [memory_param['mailbox_size']]
        

        self.use_async = self.config.use_async_prefetch


        self.use_valid_edge = self.config.use_valid_edge
        
        if (self.use_valid_edge):
            self.unexpired_edge_map = None
            self.unexpired_edge_feats = None


            expire_info = None
            file_path = f'/DOLPHIN/preprocessing/expire-{self.batch_size}.json'
            with open(file_path, 'r', encoding='utf-8') as file:
                expire_info = json.load(file)
            if (self.prefetch_conn is not None and (self.config.use_valid_edge and not self.config.use_disk)):
                self.prefetch_conn.send(('init_valid_edge', (expire_info[d], self.edge_feat_dim, self.total_edge_num, (self.path, self.batch_size, self.sampler.fan_nums))))
                self.prefetch_conn.recv()
            

    def init_share_tensor(self):
        node_num = self.share_node_num
        edge_num = self.share_edge_num
        node_feat_dim = self.node_feat_dim
        edge_feat_dim = self.edge_feat_dim
        

        part_node_map = torch.zeros(node_num, dtype = torch.int32).share_memory_()
        node_feats = torch.zeros((node_num, node_feat_dim), dtype = getattr(torch, self.node_feat_type)).share_memory_()

        part_edge_map = torch.zeros(edge_num, dtype = torch.int32).share_memory_()
        edge_feats = torch.zeros((edge_num, edge_feat_dim), dtype = getattr(torch, self.edge_feat_type)).share_memory_()

        if (self.use_memory):
            mem_dim = self.memory_param['dim_out']
            mailbox_size = self.memory_param['mailbox_size']
            part_memory = torch.zeros((node_num, mem_dim), dtype = torch.float32).share_memory_()
            part_memory_ts = torch.zeros(node_num, dtype = torch.float32).share_memory_()
            part_mailbox = torch.zeros((node_num, mailbox_size, 2 * mem_dim + edge_feat_dim), dtype = torch.float32).share_memory_()
            part_mailbox_ts = torch.zeros((node_num, mailbox_size), dtype = torch.float32).share_memory_()
        else:
            part_memory = None
            part_memory_ts = None
            part_mailbox = None
            part_mailbox_ts = None

        pre_same_nodes = torch.zeros(node_num, dtype = torch.bool).share_memory_()
        cur_same_nodes = torch.zeros(node_num, dtype = torch.bool).share_memory_()

        shared_tensor = (part_node_map, node_feats, part_edge_map, edge_feats, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts, pre_same_nodes, cur_same_nodes)

        shared_node_d_ind = torch.zeros(node_num, dtype = torch.int32).share_memory_()
        shared_edge_d_ind = torch.zeros(edge_num, dtype = torch.int32).share_memory_()

        shared_tensor = (*shared_tensor, shared_node_d_ind, shared_edge_d_ind)
        self.share_node_d_ind = shared_node_d_ind
        self.share_edge_d_ind = shared_edge_d_ind

        
        shared_ret_len = torch.zeros(len(shared_tensor), dtype = torch.int32).share_memory_()

        shared_tensor = (*shared_tensor, shared_ret_len)
        
        self.share_part_node_map = part_node_map
        self.share_part_node_feats = node_feats
        self.share_part_edge_map = part_edge_map
        self.share_part_edge_feats = edge_feats
        self.share_part_memory = part_memory
        self.share_part_memory_ts = part_memory_ts
        self.share_part_mailbox = part_mailbox
        self.share_part_mailbox_ts = part_mailbox_ts
        self.share_pre_same_nodes = pre_same_nodes
        self.share_cur_same_nodes = cur_same_nodes
        self.shared_ret_len = shared_ret_len
        
        self.share_tmp_tensor = torch.zeros(self.tmp_tensor_num).share_memory_()
        shared_tensor = (*shared_tensor, self.share_tmp_tensor)

        

        self.prefetch_conn.send(('init_share_tensor', (shared_tensor,)))
        self.prefetch_conn.recv()

        


    def init_feat(self, node_feats, edge_feats):
        self.node_feats = node_feats
        self.edge_feats = edge_feats

    def init_memory(self, memory, memory_ts, mailbox, mailbox_ts):
        self.memory = memory
        self.memory_ts = memory_ts
        self.mailbox = mailbox
        self.mailbox_ts = mailbox_ts

    def select_index(self, name, indices):
        self_v = getattr(self,name, None)
        if (self_v is not None):
            return self_v[indices]
        elif (name in ['node_feats', 'edge_feats'] and self.use_disk):
            result = loadBinDisk(getattr(self, f'{name}_path'), indices)
        else:
            self.prefetch_conn.send(('select_index', (name, indices.cpu())))
            data = self.prefetch_conn.recv()
            dim, shape = data
            result = self.share_tmp_tensor[:shape].reshape(dim)
        return result
        
    def update_index(self, name, indices, value):
        self_v = getattr(self,name, None)
        if (self_v is not None):
            self_v[indices] = value
        else:
            dim = value.shape
            value = value.reshape(-1)
            shape = value.shape[0]

            self.share_tmp_tensor = self.share_tmp_tensor.to(value.dtype)
            self.share_tmp_tensor[:shape] = value

            self.prefetch_conn.send(('update_index', (name, indices, (dim, shape))))
            self.prefetch_conn.recv()

    def reset_time(self):
        assert self.err_num == 0

    def print_time(self):
        pass


    def reset(self):
        self.part_edge_feats = None
        self.part_edge_map = None
        self.part_node_feats = None
        self.part_node_map = None

        self.part_memory = None
        self.part_memory_ts = None
        self.part_mailbox = None
        self.part_mailbox_ts = None

        self.cur_block = 0

        self.time_load = 0
        self.time_analyze = 0
        self.time_presample = 0
        self.time_pre_neg_sample = 0
        self.time_exec_mem = 0
        self.time_neg_analyze = 0
        self.time_refresh = 0
        self.time_async = 0

        if (self.mem_detection):
            self.det_memory.zero_()
            self.det_memory_ts.zero_()
            self.det_mailbox.zero_()
            self.det_mailbox_ts.zero_()

        if (self.prefetch_conn is not None and (self.config.use_valid_edge and not self.config.use_disk)):
            self.prefetch_conn.send(('reset_valid_edge', ()))
            self.prefetch_conn.recv()
        


    def get_mailbox(self, nid):

        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)

        return self.part_mailbox[table2].reshape(nid.shape[0], -1).cuda()
       

    def get_mails(self, nid):
        
        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)
        if (not self.transfer_cpu):
            return[self.part_memory[table2], self.part_memory_ts[table2],self.part_mailbox[table2],self.part_mailbox_ts[table2]]
        else:
            return[self.part_memory[table2].cpu(), self.part_memory_ts[table2].cpu(),self.part_mailbox[table2].cpu(),self.part_mailbox_ts[table2].cpu()]


    def input_mails(self, b):
        
        nid = b.srcdata['ID']
        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)


        b.srcdata['mem'] = self.part_memory[table2].cuda()
        b.srcdata['mem_ts'] = self.part_memory_ts[table2].cuda()
        b.srcdata['mem_input'] = self.part_mailbox[table2].reshape(b.srcdata['ID'].shape[0], -1).cuda()
        b.srcdata['mail_ts'] = self.part_mailbox_ts[table2].cuda()

        if(self.mem_detection):
            err_num = torch.sum(b.srcdata['mem'].cpu() != self.det_memory[nid.long().cpu()])
            err_num += torch.sum(b.srcdata['mem_ts'].cpu() != self.det_memory_ts[nid.long().cpu()])
            err_num += torch.sum(b.srcdata['mem_input'].cpu() != self.det_mailbox[nid.long().cpu()].reshape(b.srcdata['ID'].shape[0], -1))
            err_num += torch.sum(b.srcdata['mail_ts'].cpu() != self.det_mailbox_ts[nid.long().cpu()])
            if (err_num + torch.nonzero(table2 == -1).shape[0]):
                raise BufferError("There has been an incident where the internal state of the buffer is inconsistent with the non-cached scenario!")

    def update_mailbox(self, nid, mail, mail_ts, n_ptr = None):
        nid = nid.cuda()
        if (n_ptr is not None):
            n_ptr = n_ptr.cuda()
        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)

        if (n_ptr is None):
            n_ptr = 0
            
        self.part_mailbox[table2, n_ptr] = mail
        self.part_mailbox_ts[table2, n_ptr] = mail_ts

        if (self.mem_detection):
            self.det_mailbox[nid.long().cpu(), n_ptr] = mail.cpu()
            self.det_mailbox_ts[nid.long().cpu(), n_ptr] = mail_ts.cpu()

    def update_memory(self, nid, memory, ts):
        table1 = torch.zeros_like(self.part_memory_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_memory_map, nid, table1, table2)
        table2 = table2.to(torch.int64)


        self.part_memory[table2] = memory
        self.part_memory_ts[table2] = ts

        if (self.mem_detection):
            self.det_memory[nid.long().cpu()] = memory.cpu()
            self.det_memory_ts[nid.long().cpu()] = ts.cpu()


    def get_e_feat(self, eid, compute_index_time = False):
        index_s = time.time()
        table1 = torch.zeros_like(self.part_edge_map) - 1
        table2 = torch.zeros_like(eid) - 1
        dgl.findSameIndex(self.part_edge_map, eid, table1, table2)
        if (compute_index_time):
            print(f"get_e_feat index time: {time.time() - index_s:.4f}s")

        res = self.part_edge_feats[table2.to(torch.int64)]

        return res.cuda()

    def get_n_feat(self, nid):
        start = time.time()
        table1 = torch.zeros_like(self.part_node_map) - 1
        table2 = torch.zeros_like(nid) - 1
        dgl.findSameIndex(self.part_node_map, nid, table1, table2)

       
        res = self.part_node_feats[table2.to(torch.int64)]
        if (self.err_detection):
            err_num = torch.sum(res.cpu() != self.det_node_feats[nid.long().cpu()])
            if (err_num + torch.nonzero(table2 == -1).shape[0]):
                print(f"error! : {err_num}")
                
        
        
        return res.cuda()

    def pre_fetch(self, block_num, memory_info):
        
        print(block_num)
        
        neg_nodes, neg_eids = self.pre_neg_sample(block_num)
        neg_nodes, _ = torch.sort(neg_nodes)
        neg_eids, _ = torch.sort(neg_eids)

        
        path = self.path
        pos_edge_feats = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_edge_feat.pt')
        pos_edge_map = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_edge_map.pt').cuda()
        pos_node_feats = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_node_feat.pt')
        pos_node_map = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{block_num}_node_map.pt').cuda()
        
        table1 = torch.zeros_like(neg_nodes) - 1
        table2 = torch.zeros_like(pos_node_map) - 1
        
        dgl.findSameNode(neg_nodes, pos_node_map, table1, table2)
        neg_nodes, pos_node_map, table1, table2 = neg_nodes.cpu(), pos_node_map.cpu(), table1.cpu(), table2.cpu()
        dis_ind = table1 == -1
        dis_neg_nodes = neg_nodes[dis_ind]
        neg_node_feats = self.node_feats[dis_neg_nodes.to(torch.int64)]

        pos_node_map = torch.cat((pos_node_map, dis_neg_nodes))
        pos_node_map,indices = torch.sort(pos_node_map)
        node_feats = torch.cat((pos_node_feats, neg_node_feats))
        node_feats = node_feats[indices]

        table1 = torch.zeros_like(neg_eids) - 1
        table2 = torch.zeros_like(pos_edge_map) - 1
        dgl.findSameNode(neg_eids, pos_edge_map, table1, table2)
        neg_eids, pos_edge_map, table1, table2 = neg_eids.cpu(), pos_edge_map.cpu(), table1.cpu(), table2.cpu()
        dis_ind = table1 == -1
        dis_neg_eids = neg_eids[dis_ind]
        dis_neg_eids_feat = self.edge_feats[dis_neg_eids.to(torch.int64)]

        pos_edge_map = torch.cat((pos_edge_map, dis_neg_eids))
        pos_edge_map,indices = torch.sort(pos_edge_map)
        edge_feats = torch.cat((pos_edge_feats, dis_neg_eids_feat))
        edge_feats = edge_feats[indices]

        
        if (memory_info):
            part_memory_map, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = memory_info
            part_map = part_memory_map.cpu().clone().long()
            self.memory[part_map] = part_memory.cpu().clone()
            self.memory_ts[part_map] = part_memory_ts.cpu().clone()
            self.mailbox[part_map] = part_mailbox.cpu().clone()
            self.mailbox_ts[part_map] = part_mailbox_ts.cpu().clone()

        nodes = pos_node_map.long()
        part_memory = self.memory[nodes]
        part_memory_ts = self.memory_ts[nodes]
        part_mailbox = self.mailbox[nodes]
        part_mailbox_ts = self.mailbox_ts[nodes]
       
        pre_same_nodes = torch.isin(self.part_node_map.cpu(), pos_node_map)
        cur_same_nodes = torch.isin(pos_node_map, self.part_node_map.cpu())

        print(f"pre fetch over...")
        self.preFetchDataCache.put({'node_info': [pos_node_map, node_feats], 'edge_info': [pos_edge_map, edge_feats],\
                                     'memory_info': [part_memory, part_memory_ts, part_mailbox, part_mailbox_ts],\
                                        'memory_update_info': [pre_same_nodes, cur_same_nodes]})



    def move_to_gpu(self, datas, flag=False, use_pin = False):
        res = []
        for data in datas:
            try:
                if (use_pin):
                    data = data.pin_memory()
                else:
                    data = data.cuda()
                res.append(data)
            except RuntimeError as e:
                if (flag):
                    print(e)
                    raise e
                emptyCache()
                res.append(self.move_to_gpu([data], flag=True, use_pin=use_pin)[0])

        return res


    def cal_io_mem(self, tensors):
        res = 0
        for tensor in tensors:
            if (tensor is not None):
                # print(tensor.numel())
                res += tensor.numel() * 4
        return res
    def prefetch_after(self):

        io_start = time.time()
        node_num = self.shared_ret_len[0]
        edge_num = self.shared_ret_len[2]
        pre_num = self.shared_ret_len[8]
        cur_num = self.shared_ret_len[9]

        
        use_pin = hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory

        allo1 = cuda_GB()

        node_d_ind_num = self.shared_ret_len[10].cuda()
        edge_d_ind_num = self.shared_ret_len[11].cuda()
        node_d_ind, edge_d_ind = self.share_node_d_ind[:node_d_ind_num].long().cuda(), self.share_edge_d_ind[:edge_d_ind_num].long().cuda()
        
        
        if (self.edge_feat_dim > 0):
            part_edge_map = self.move_to_gpu([self.share_part_edge_map[:edge_num]])[0]
            edge_d_feat = self.get_e_feat(part_edge_map[edge_d_ind])
            self.part_edge_feats = None
            part_edge_feats = self.move_to_gpu([self.share_part_edge_feats[:edge_num]])[0]
            part_edge_feats[edge_d_ind] = edge_d_feat
            self.io_mem += (part_edge_feats.numel() - edge_d_ind.shape[0] * part_edge_feats.shape[1]) * 4
            self.part_edge_map, self.part_edge_feats = part_edge_map, part_edge_feats

        part_node_map, part_node_feats = self.share_part_node_map[:node_num].cuda(), self.share_part_node_feats[:node_num].cuda()
        part_node_feats[node_d_ind] = self.get_n_feat(part_node_map[node_d_ind])
        self.io_mem += (part_node_feats.numel() - node_d_ind.shape[0] * part_node_feats.shape[1]) * 4
        self.part_node_map, self.part_node_feats = part_node_map, part_node_feats



        self.part_memory_map = self.part_node_map

        if (self.use_memory):
            part_memory, part_memory_ts, part_mailbox, part_mailbox_ts \
                = self.share_part_memory[:node_num], self.share_part_memory_ts[:node_num], self.share_part_mailbox[:node_num], self.share_part_mailbox_ts[:node_num]
            
            part_memory, part_memory_ts, part_mailbox, part_mailbox_ts = self.move_to_gpu([part_memory, part_memory_ts, part_mailbox, part_mailbox_ts])
        
            pre_same_nodes, cur_same_node = self.share_pre_same_nodes[:pre_num], self.share_cur_same_nodes[:cur_num]
            pre_same_nodes, cur_same_node = pre_same_nodes.cuda(), cur_same_node.cuda()

            part_memory[cur_same_node] = self.part_memory[pre_same_nodes]
            part_memory_ts[cur_same_node] = self.part_memory_ts[pre_same_nodes]
            part_mailbox[cur_same_node] = self.part_mailbox[pre_same_nodes]
            part_mailbox_ts[cur_same_node] = self.part_mailbox_ts[pre_same_nodes]

            same_num = torch.sum(cur_same_node)
            self.io_mem += (part_memory.numel() - same_num * part_memory.shape[1]) * 4
            self.io_mem += (part_memory_ts.numel() - same_num) * 4
            self.io_mem += (part_mailbox.numel() - same_num * part_mailbox.shape[1]) * 4
            self.io_mem += (part_mailbox_ts.numel() - same_num) * 4
            self.part_memory, self.part_memory_ts,self.part_mailbox,self.part_mailbox_ts = part_memory,part_memory_ts,part_mailbox,part_mailbox_ts

        self.io_time += time.time() - io_start
        # self.io_mem += self.cal_io_mem([self.part_memory,self.part_memory_ts,self.part_mailbox,self.part_mailbox_ts,self.part_edge_map, self.part_edge_feats,self.part_node_map, self.part_node_feats])

        allo2 = cuda_GB()
        # print(f"prefetch_after: {allo1} -> {allo2}")

    def run_batch(self, left,test_block = 0):
        
        if (test_block > 0):
            self.first_test = True
        else:
            self.first_test = False

        
        self.cur_batch = int(torch.div(left, self.block_size, rounding_mode='trunc').item())
        # if (cur_batch * self.batch_size in self.bucket_config['bucket_ptr'] or test_block > 0):
        if (left % self.block_size == 0 or test_block > 0):
        
        
            start = ((self.cur_block - 1) * self.batch_num) * self.train_batch_size
            end = min(self.train_edge_end, (((self.cur_block - 1) * self.batch_num) + self.batch_num) * self.train_batch_size)

            # start = self.bucket_ptr[self.cur_block - 1]
            # end = min(self.train_edge_end, self.bucket_ptr[self.cur_block - 1 + 1])
            if (not self.use_memory):
                memory_info = None
            else:
                if ((self.part_node_map is None or self.config.model_eval)):
                    memory_info = (self.part_node_map, self.part_memory, self.part_memory_ts, self.part_mailbox, self.part_mailbox_ts)
                else:
                    root_nodes = (torch.cat((self.datas['src'][start:end], self.datas['dst'][start:end]))).cuda()
                    root_nodes = torch.unique(root_nodes)

                    next_start = ((self.cur_block) * self.batch_num) * self.train_batch_size
                    next_end = min(self.train_edge_end, (((self.cur_block) * self.batch_num) + self.batch_num) * self.train_batch_size)
                    next_root_nodes = (torch.cat((self.datas['src'][next_start:next_end], self.datas['dst'][next_start:next_end]))).cuda()
                    next_root_nodes = torch.unique(next_root_nodes)
                    root_nodes = root_nodes[torch.isin(root_nodes, next_root_nodes, invert=True, assume_unique=True)]
                    if (self.transfer_cpu):
                        memory_info = (root_nodes.cpu(), *self.get_mails(root_nodes))
                    else:
                        memory_info = (root_nodes, *self.get_mails(root_nodes))

            # memory_info = (self.part_node_map, self.part_memory, self.part_memory_ts, self.part_mailbox, self.part_mailbox_ts)
            if (self.cur_block == 0):
                time_A = time.time()
                print(f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_node_feat_0.pt')
                bucket_cache_node_feat = loadBin(self.path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_node_feat_0.pt').share_memory_()
                bucket_cache_edge_feat = None
                if (self.has_ef):
                    print(self.path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_edge_feat_0.pt')
                    bucket_cache_edge_feat = loadBin(self.path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_edge_feat_0.pt').share_memory_()
                shared_tensor = [bucket_cache_node_feat, bucket_cache_edge_feat]
                self.prefetch_conn.send(('init_bucket_cache', (shared_tensor, self.path, f'/part-{self.batch_size}-{self.sampler.fan_nums}')))#
                self.prefetch_conn.recv()

            if (self.cur_block == 0 or test_block > 0):
                self.cur_block = test_block
                time_first = time.time()
                time_load_s = time.time()
                if (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory):
                    self.load_part_pin(self.cur_batch // self.batch_num)
                else:
                    self.load_part(self.cur_block)
                self.time_load += time.time() - time_load_s

                time_ana_s = time.time()
                self.analyze()
                self.time_analyze += time.time() - time_ana_s
            
                
            else:
              
                if (self.use_async):
                    time_load_s = time.time()
                    flag = self.prefetch_only_conn.recv()
                    self.time_async += time.time() - time_load_s
                else:
                    
                    time_load_s = time.time()
                    self.prefetch_conn.send(('pre_fetch', (self.cur_block,self.memory_info,self.neg_info, self.part_node_map.cpu(), self.part_edge_map.cpu(),\
                                        (self.path, self.batch_size, self.sampler.fan_nums))))
                    flag = self.prefetch_only_conn.recv()
                if (flag is not None and 'extension' in flag):
                    print(f"{flag} share tensor error...")
                time_s = time.time()
                self.prefetch_after()
                #print("IO",time.time()-time_s)
                self.time_load += time.time() - time_load_s

            time_ana_s = time.time()
            self.neg_sample_nodes = self.neg_sample_nodes_async
            


            neg_info = self.pre_neg_sample(self.cur_block + 1)
            if (neg_info is not None and self.transfer_cpu):
                neg_info = [neg_info[0].cpu(), neg_info[1].cpu()]
            if (self.use_async):
                if (neg_info is not None):
                    if (self.transfer_cpu):
                        
                        self.prefetch_conn.send(('pre_fetch', (self.cur_block + 1,memory_info,neg_info, self.part_node_map.cpu(), self.part_edge_map.cpu(), (self.path, self.batch_size, self.sampler.fan_nums))))
                    else:
                        time_s = time.time()
                        self.prefetch_conn.send(('pre_fetch', (self.cur_block + 1,memory_info,neg_info, self.part_node_map, self.part_edge_map,\
                                        (self.path, self.batch_size, self.sampler.fan_nums))))
            else:
                if (neg_info is not None):
                    self.memory_info = memory_info
                    self.neg_info = neg_info
                
            self.time_analyze += time.time() - time_ana_s



            self.cur_block += 1
            


    def load_part(self, part_num):
        path = self.path
        if (self.edge_feat_dim > 0):
            self.part_edge_feats = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_feat.pt').cuda()
            self.part_edge_map = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_map.pt').cuda()
        if (self.node_feat_dim > 0):
            self.part_node_feats = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_feat.pt').cuda()
        self.part_node_map = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_map.pt').cuda()

    def load_part_pin(self, part_num):
        path = self.path
        self.part_edge_feats = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_feat.pt').pin_memory()
        self.part_edge_map = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_edge_map.pt').cuda()
        self.part_node_feats = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_feat.pt').pin_memory()
        self.part_node_map = loadBin(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{part_num}_node_map.pt').cuda()

    def pre_sample(self):
        time_presample_s = time.time()
        emptyCache()
        start = self.cur_batch * self.train_batch_size
        end = min(self.train_edge_end, (self.cur_batch + self.batch_num) * self.train_batch_size)

        src = self.datas['src'][start: end]
        dst = self.datas['dst'][start: end]
        times = self.datas['time'][start: end]

        self.neg_sample_nodes = self.neg_sampler.sample(src.shape[0])
        root_nodes = torch.from_numpy(np.concatenate([src, dst, self.neg_sample_nodes]).astype(np.int32)).cuda()
        root_ts = torch.from_numpy(np.concatenate([times, times, times]).astype(np.float32)).cuda()


        ret_list = self.sampler.sample_layer(root_nodes, root_ts)
        src,dst,outts,outeid,root_nodes,root_ts,dts = ret_list[-1]

        mask = src > -1
        src = src[mask]
        dst = dst[mask]
        del dst, mask, outts, outeid, root_ts, ret_list
        emptyCache()
        nodes = torch.cat((src, root_nodes))
        nodes = torch.unique(nodes)
        
        self.time_presample += time.time() - time_presample_s
        return nodes

    def pre_neg_sample(self, block_num):
        time_pre_neg_s = time.time()
        cur_start = 0
        cur_end = 0
        
        
        start = (block_num * self.batch_num) * self.train_batch_size
        end = min(self.train_edge_end, ((block_num * self.batch_num) + self.batch_num) * self.train_batch_size)
        # start = self.bucket_ptr[block_num]
        # end = min(self.train_edge_end, self.bucket_ptr[block_num + 1])
        if (self.mode == 'test_train'):
            end = min(self.val_edge_end, ((block_num * self.batch_num) + self.batch_num) * self.train_batch_size)
        if (self.mode == 'test' or self.mode == 'val'):
            end = min(self.test_edge_end, ((block_num * self.batch_num) + self.batch_num) * self.train_batch_size)

        if(start >= end):
            return None

        times = self.datas['time'][start: end]


        if (self.first_test):
            self.neg_sample_nodes_async = self.test_neg_sampler.sample_test(self.val_edge_end, end)
            self.first_test = False
        else:
            if (self.mode == 'train'):
                self.neg_sample_nodes_async = self.neg_sampler.sample((times.shape[0]))

            else:
                if (self.mode != 'test' and start < self.val_edge_end and end < self.val_edge_end):
                    self.neg_sample_nodes_async = self.test_neg_sampler.sample((times.shape[0]))
                else:
                    if (start >= self.val_edge_end):
                        self.neg_sample_nodes_async = self.test_neg_sampler.sample_test(start, end)
                    elif (end >= self.val_edge_end):
                        self.neg_sample_nodes_async = np.zeros(end - start, dtype = np.int32)
                        mid_ind = self.val_edge_end - start
                        self.neg_sample_nodes_async[:mid_ind] = self.test_neg_sampler.sample(mid_ind)
                        self.neg_sample_nodes_async[mid_ind:] = self.test_neg_sampler.sample_test(self.val_edge_end, end)
            

                
        root_nodes = torch.from_numpy(np.concatenate([self.neg_sample_nodes_async]).astype(np.int32)).cuda()
        root_ts = torch.from_numpy(np.concatenate([times]).astype(np.float32)).cuda()


        ret_list = self.sampler.sample_layer(root_nodes, root_ts)
        src,dst,outts,outeid,root_nodes,root_ts,dts = ret_list[-1]

        mask = src > -1
        src = src[mask]
        dst = dst[mask]
        nodes = torch.cat((src, root_nodes))
        nodes = torch.unique(nodes)

        eid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
        for ret in ret_list:
            src,dst,outts,outeid,root_nodes,root_ts,dts = ret
            eid = outeid[outeid > -1]

            cur_eid = torch.unique(eid)
            eid_uni = torch.cat((eid_uni, cur_eid))
            eid_uni = torch.unique(eid_uni)
        
        self.time_pre_neg_sample += time.time() - time_pre_neg_s

        return nodes, eid_uni

    def refresh_memory(self):
        time_refresh_s = time.time()
        if (not self.use_memory):
            return
        
        if (self.part_memory) != None:
            part_map = self.part_memory_map.long().cpu()
            self.update_index('memory', part_map, self.part_memory.cpu())
            self.update_index('memory_ts', part_map, self.part_memory_ts.cpu())
            self.update_index('mailbox', part_map, self.part_mailbox.cpu())
            self.update_index('mailbox_ts', part_map, self.part_mailbox_ts.cpu())
        self.time_refresh += time.time() - time_refresh_s

    
    def analyze_mem(self, nodes):
        time_exec_mem_s = time.time()
        self.refresh_memory()

        if (self.use_disk):
            self.part_memory = torch.zeros([nodes.shape[0]] + self.memory_shape, dtype = torch.float32, device = 'cuda:0')
            self.part_memory_ts = torch.zeros([nodes.shape[0]], dtype = torch.float32, device = 'cuda:0')
            self.part_mailbox = torch.zeros([nodes.shape[0]] + self.mailbox_shape, dtype = torch.float32, device = 'cuda:0')
            self.part_mailbox_ts = torch.zeros([nodes.shape[0]] + self.mailbox_ts_shape, dtype = torch.float32, device = 'cuda:0')
            
        elif (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory and False):
            self.part_memory = self.select_index('memory',nodes).pin_memory()
            self.part_memory_ts = self.select_index('memory_ts',nodes).pin_memory()
            self.part_mailbox = self.select_index('mailbox',nodes).pin_memory()
            self.part_mailbox_ts = self.select_index('mailbox_ts',nodes).pin_memory()
        else:
            self.part_memory = self.select_index('memory',nodes).cuda()
            self.part_memory_ts = self.select_index('memory_ts',nodes).cuda()
            self.part_mailbox = self.select_index('mailbox',nodes).cuda()
            self.part_mailbox_ts = self.select_index('mailbox_ts',nodes).cuda()

        self.part_memory_map = nodes.to(torch.int32).cuda()
        self.time_exec_mem += time.time() - time_exec_mem_s

    def analyze_async(self):

        
        aasdas = 1

    def analyze(self):



        
        neg_nodes,neg_eids = self.pre_neg_sample(self.cur_batch // self.batch_num)

        time_neg_analyze_s = time.time()
        neg_nodes, _ = torch.sort(neg_nodes)
        neg_eids,_ = torch.sort(neg_eids)

        table1 = torch.zeros_like(neg_nodes) - 1
        table2 = torch.zeros_like(self.part_node_map) - 1
        dgl.findSameNode(neg_nodes, self.part_node_map, table1, table2)
        dis_ind = table1 == -1
        dis_neg_nodes = neg_nodes[dis_ind]
        if (self.node_feat_dim > 0):
            dis_neg_nodes_feat = self.select_index('node_feats', dis_neg_nodes.cpu().to(torch.int64)).cuda()
            if (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory):
                dis_neg_nodes_feat = dis_neg_nodes_feat.cpu().pin_memory()

        self.part_node_map = torch.cat((self.part_node_map, dis_neg_nodes))
        self.part_node_map,indices = torch.sort(self.part_node_map)
        if (self.node_feat_dim > 0):
            self.part_node_feats = torch.cat((self.part_node_feats, dis_neg_nodes_feat))
            self.part_node_feats = self.part_node_feats[indices]



        
        if (self.edge_feat_dim > 0):
            table1 = torch.zeros_like(neg_eids) - 1
            table2 = torch.zeros_like(self.part_edge_map) - 1
            dgl.findSameNode(neg_eids, self.part_edge_map, table1, table2)
            dis_ind = table1 == -1
            dis_neg_eids = neg_eids[dis_ind]
            dis_neg_eids_feat = self.select_index('edge_feats',dis_neg_eids.cpu().to(torch.int64)).cuda()
            if (hasattr(self.config, 'use_pin_memory') and self.config.use_pin_memory):
                dis_neg_eids_feat = dis_neg_eids_feat.cpu().pin_memory()

        
            self.part_edge_map = torch.cat((self.part_edge_map, dis_neg_eids))
            self.part_edge_map,indices = torch.sort(self.part_edge_map)
        if (self.edge_feat_dim > 0):
            self.part_edge_feats = torch.cat((self.part_edge_feats, dis_neg_eids_feat))
            self.part_edge_feats = self.part_edge_feats[indices]

        self.time_neg_analyze += time.time() - time_neg_analyze_s


        nodes = self.part_node_map.long()
        
        if (self.use_memory):
            self.analyze_mem(nodes)


    
    def gen_part(self, mode = '', incre = False, save_feat = True, use_unique = True):
        
        d = self.d
        path = self.path
        
        df = self.df
        batch_size = self.batch_size
        if (mode == ''):
            df_start = 0
            df_end = len(df)
        if (mode == 'train'):
            df_start = 0
            df_end = self.train_edge_end
        elif (mode == 'val'):
            df_start = self.train_edge_end
            df_end = self.val_edge_end
        elif(mode == 'test'):
            df_start = self.val_edge_end
            df_end = len(df)
            
        group_indexes = np.array(df[df_start:df_end].index // batch_size)
        group_indexes -= group_indexes[0]
        left, right = df_start, df_start
        batch_num = 0

        pre_eid, pre_nid = None, None
        while True:
        # for batch_num, rows in df[df_start:df_end].groupby(group_indexes):
            # emptyCache()
            right += batch_size
            right = min(df_end, right)
            if (left >= right):
                break
            rows = df[left:right]
            root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
            root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()

            # eids = torch.from_numpy(rows['Unnamed: 0']).to(torch.int32).cuda()
            start = time.time()
            ret_list = self.sampler.sample_layer(root_nodes, root_ts)
            eid_uni = torch.from_numpy(rows['Unnamed: 0'].values).to(torch.int32).cuda()
            saveBin(torch.tensor([left, right], dtype = torch.int32).cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_bound.bin')
            root_eids_num = eid_uni.shape[0]


            nid_uni = torch.unique(root_nodes).to(torch.int32).cuda()
            # nid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')

            for ret in ret_list:
        
                src,dst,outts,outeid,root_nodes,root_ts,dts = ret
                eid = outeid[outeid > -1]

                cur_eid = torch.unique(eid)
                eid_uni = torch.cat((cur_eid, eid_uni))
                eid_uni = torch.unique(eid_uni)
            
            ret = ret_list[-1]
            src,dst,outts,outeid,root_nodes,root_ts,dts = ret
            del ret_list
            del outts, outeid, root_ts, dst
            # emptyCache()

            mask = src > -1
            src = src[mask]
            nid_uni = torch.cat((src, root_nodes))
            nid_uni = torch.unique(nid_uni)

            
            sample_time = time.time() - start
            save_time = 0

            if (not incre or pre_eid is None):
                eid_uni,_ = torch.sort(eid_uni)
                if ((self.edge_feats is not None and self.edge_feats.shape[0] > 0) or not (save_feat)):
                    cur_edge_feat = self.select_index('edge_feats',eid_uni.to(torch.int64))
                    save_start = time.time()
                    saveBin(cur_edge_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_feat{mode}.pt')
                    save_time += time.time() - save_start
                    flush_saveBin_conf()

                save_start = time.time()
                saveBin(eid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_map{mode}.pt')
                save_time += time.time() - save_start

                nid_uni,_ = torch.sort(nid_uni)
                if ((self.node_feats is not None and self.node_feats.shape[0] > 0) or not (save_feat)):
                    cur_node_feat = self.select_index('node_feats',nid_uni.to(torch.int64))
                    save_start = time.time()
                    saveBin(cur_node_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_feat{mode}.pt')
                    save_time += time.time() - save_start
                    flush_saveBin_conf()

                save_start = time.time()
                saveBin(nid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_map{mode}.pt')
                save_time += time.time() - save_start
            else:
                eid_uni,_ = torch.sort(eid_uni)
                eid_incre_mask = torch.isin(eid_uni, pre_eid, assume_unique=True, invert = True)
                cur_eid = eid_uni[eid_incre_mask]

                if (self.edge_feats is not None and self.edge_feats.shape[0] > 0 and save_feat):
                    cur_edge_feat = self.select_index('edge_feats',cur_eid[:cur_eid.shape[0] - root_eids_num].to(torch.int64))
                    save_start = time.time()
                    saveBin(cur_edge_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_feat{mode}_incre.pt')
                    save_time += time.time() - save_start
                save_start = time.time()
                saveBin(eid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_map{mode}.pt')
                saveBin(eid_incre_mask.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_map{mode}_incre_mask.pt')
                save_time += time.time() - save_start

                nid_uni,_ = torch.sort(nid_uni)
                nid_incre_mask = torch.isin(nid_uni, pre_nid, assume_unique=True, invert = True)
                cur_nid = nid_uni[nid_incre_mask]
                if (self.node_feats is not None and self.node_feats.shape[0] > 0 and save_feat):
                    cur_node_feat = self.select_index('node_feats',cur_nid.to(torch.int64))
                    save_start = time.time()
                    saveBin(cur_node_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_feat{mode}_incre.pt')
                    save_time += time.time() - save_start
                save_start = time.time()
                saveBin(nid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_map{mode}.pt')
                saveBin(nid_incre_mask.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_map{mode}_incre_mask.pt')
                save_time += time.time() - save_start
            # mfgs = sampler.gen_mfgs(ret_list)
            
            print(f"edges: {eid_uni.shape} nodes:{nid_uni.shape} batch: {batch_num} totalTime:{time.time() - start:.7f}s sampleTime: {sample_time:.2f}s saveTime: {save_time:.2f}s")
            # print(f"edges: {cur_edge_feat.shape} nodes:{cur_node_feat.shape} batch: {batch_num} totalTime:{time.time() - start:.7f}s sampleTime: {sample_time:.2f}s saveTime: {save_time:.2f}s")
            if (incre):
                pre_eid = eid_uni
                pre_nid = nid_uni
            del root_nodes,eid_uni,nid_uni,src,mask,eid,_
            

            left = right
            batch_num += 1



    def stream_extract(self, feat_path, save_paths, window_size, his, his_max, feat_len, np_type):
        mask_time = 0
        ind_time = 0
        max_feat_len = max(his_max).item() + 1
        # print(his_max)
        start = 0
        end = 0

        time_start = time.time()
            

        while True:
            end += window_size
            end = min(end, max_feat_len)
            print(f"end: {end} max_feat_len: {max_feat_len}")
            if (start >= end):
                break

            row_data = np.fromfile(feat_path, dtype=np_type, offset=start * feat_len, count=(end - start) * feat_len)
            row_data = row_data.reshape(-1, feat_len)

            for i, tensor in enumerate(his):
                mask_s = time.time()  
                mask = torch.bitwise_and(tensor >= start, tensor < end)
                mask_time += time.time() - mask_s

                ind_s = time.time()
                cur_ind = tensor[mask]
                ind_time += time.time() - ind_s
                print(cur_ind)
                cur_data = torch.from_numpy(row_data[cur_ind - start].reshape(-1, feat_len))
                cur_save_path = save_paths[i]

                    
                if (i > 0):
                    cur_save_path = cur_save_path.replace('.bin', '_incre.bin')
                saveBin(cur_data, cur_save_path, addSave=(start > 0))

            del row_data
            gc.collect()
            print(f"{start}:{end} mask: {mask_time:.2f}s  ind: {ind_time:.2f}s")
            start = end
        flush_saveBin_conf()

        print(f"total time: {time.time() - time_start:.4f}s")


    def gen_part_stream(self, budget = (20 * 1024), bucket_budget = 5 * 1024 ** 3, bucket_optimal = False, bucket_cache = False, cache_budget = 4 * 1024 ** 3):

        has_ef = self.edge_feat_dim > 0
        has_nf = self.node_feat_dim > 0
        threshold = 80000000 

        if (self.d == 'BITCOIN'):
            node_cache_budget = cache_budget * (1/3.5)
            edge_cache_budget = cache_budget * (2.5/2.5)
        elif (self.d == 'MAG'):
            node_cache_budget = cache_budget
        else:
            node_cache_budget = cache_budget * (1/4) 
            edge_cache_budget = cache_budget * (3/4) 
        node_idx_budget = int(node_cache_budget / self.node_feat_dim / 4)
        edge_idx_budget = int(edge_cache_budget / self.edge_feat_dim / 4) if self.has_ef else 0
        d = self.d
        path = self.path

        df = self.df
        batch_size = self.batch_size

        budget_byte = budget * 1024 * 1024
        node_his = []
        edge_his = []
        node_his_max = []
        edge_his_max = []
        edge_save_paths = []
        node_save_paths = []
        node_window_size = int(budget_byte * 0.9 / 4 / self.node_feat_dim)
        edge_window_size = int(budget_byte * 0.9 / 4 / self.edge_feat_dim) if self.edge_feat_dim != 0 else 0
        history_datas = []
        his_mem_threshold = 4 * 1024 ** 3  # 4GB
        his_mem_byte = 0 

        res_config = {}
        res_config['bucket_ptr'] = [0]
        res_config['max_node_num'] = 0
        res_config['max_edge_num'] = 0
        res_config['bucket_cache_node_num'] = node_idx_budget
        res_config['bucket_cache_edge_num'] = edge_idx_budget

        left, right = 0, 0
        batch_num = 0
        datas = self.datas
        pre_eid, pre_nid = None, None
        edge_end = datas['src'].shape[0]
        if (self.d == 'MAG'):
            total_src = datas['src'].pin_memory().to(torch.int32)
            total_dst = datas['dst'].pin_memory().to(torch.int32)
            total_time = datas['time'].pin_memory().to(torch.float32)
            total_eid = datas['eid'].pin_memory().to(torch.int32)
        else:
            total_src = datas['src'].cuda().to(torch.int32)
            total_dst = datas['dst'].cuda().to(torch.int32)
            total_time = datas['time'].cuda().to(torch.float32)
            total_eid = datas['eid'].cuda().to(torch.int32)

        node_feat_path = f'{path}/node_features.bin'
        edge_feat_path =  f'{path}/edge_features.bin'
        node_save_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/parti_node_feat.bin'
        edge_save_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/parti_edge_feat.bin'
        bucket_his = []
        preThresholdNum = 0
        bucket_nid_count = torch.zeros(self.total_node_num, dtype = torch.int32, device = 'cuda:0')
        bucket_eid_count = torch.zeros(self.total_edge_num if self.has_ef else 0, dtype = torch.int32, device = 'cuda:0')
        bucket_cache_start = [0] 
        bucket_cache_idx = [] 
        bucket_cache_size = 0
        pre_bucket_nid_idx = None
        pre_bucket_eid_idx = None

        while True:
            cur_bucket_allo = 0
            start = time.time()

            pre_nid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
            pre_eid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')

            # if (left >= right):
            #     break
            first_flag = True
            left_c = left
            while True:
                right += batch_size
                right = min(edge_end, right)
                if (left >= right):
                    break
                first_flag = False

                src = total_src[left: right]
                dst = total_dst[left: right]
                times = total_time[left: right]
                eid = total_eid[left: right]
                root_nodes = torch.cat((src, dst)).cuda()
                root_ts = torch.cat((times, times)).cuda()

                ret_list = self.sampler.sample_layer(root_nodes, root_ts)
                eid_uni = eid.to(torch.int32).cuda()
                nid_uni = torch.unique(root_nodes).to(torch.int32).cuda()
                root_eids_num = eid_uni.shape[0]

                for ret in ret_list:
                    src,dst,outts,outeid,root_nodes,root_ts,dts = ret
                    eid = outeid[outeid > -1]

                    cur_eid = torch.unique(eid)
                    eid_uni = torch.cat((cur_eid, eid_uni))
                    eid_uni = torch.unique(eid_uni)

                ret = ret_list[-1]
                src,dst,outts,outeid,root_nodes,root_ts,dts = ret
                del ret_list
                del outts, outeid, root_ts, dst
                # emptyCache()
                mask = src > -1
                src = src[mask]
                nid_uni = torch.cat((src, root_nodes))
                nid_uni = torch.unique(nid_uni)
                eid_uni,_ = torch.sort(eid_uni)

                nid_uni = torch.unique(torch.cat((nid_uni, pre_nid_uni)))
                eid_uni = torch.unique(torch.cat((eid_uni, pre_eid_uni)))


                if (not bucket_optimal):
                    break

                cur_bucket_allo = nid_uni.shape[0]  * self.node_feat_dim * 4
                cur_bucket_allo += eid_uni.shape[0]  * self.edge_feat_dim * 4
                cur_bucket_allo += ((2 * self.node_feat_dim + 2 * self.edge_feat_dim) * 111 ) * (right - left_c)

                if (cur_bucket_allo < bucket_budget):
                    pre_nid_uni = nid_uni.clone()
                    pre_eid_uni = eid_uni.clone()
                    left = right
                else:
                    nid_uni = pre_nid_uni
                    eid_uni = pre_eid_uni
                    break

            res_config['max_node_num'] = int(max(res_config['max_node_num'], nid_uni.shape[0] * 1.5))
            res_config['max_edge_num'] = int(max(res_config['max_edge_num'], eid_uni.shape[0] * 1.5))
            if (first_flag):
                break
            res_config['bucket_ptr'].append(right)
            if (self.has_ef):
                saveBin(torch.tensor([left, right], dtype = torch.int32).cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_bound.bin')
                saveBin(eid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_edge_map.pt')
                nid_uni,_ = torch.sort(nid_uni)
            saveBin(nid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num}_node_map.pt')

            
            
            if (math.floor(right / threshold) > preThresholdNum):
                preThresholdNum += 1

                bucket_nid_count_sort,bucket_nid_count_sort_idx = torch.sort(bucket_nid_count, descending=True)
                cache_nid = bucket_nid_count_sort_idx[:node_idx_budget]
                cache_nid_incre = cache_nid
                if (self.has_ef):
                    bucket_eid_count_sort,bucket_eid_count_sort_idx = torch.sort(bucket_eid_count, descending= True)
                    cache_eid = bucket_eid_count_sort_idx[:edge_idx_budget]
                    cache_eid_incre = cache_eid
                
                
                
                if (pre_bucket_nid_idx is not None):

                    pre_notin_cur = torch.isin(pre_bucket_nid_idx, cache_nid, invert=True, assume_unique=True)
                    cur_notin_pre = torch.isin(cache_nid, pre_bucket_nid_idx, invert=True, assume_unique=True)
                    pre_bucket_nid_idx[pre_notin_cur] = cache_nid[cur_notin_pre]
                    saveBin(pre_notin_cur.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_nid_incre_mask_{bucket_cache_start[-1]}.pt')
                    cache_nid_incre = cache_nid[cur_notin_pre]

                else:
                    pre_bucket_nid_idx = cache_nid

                if (self.has_ef):
                    if (pre_bucket_eid_idx is not None):
                        pre_notin_cur = torch.isin(pre_bucket_eid_idx, cache_eid, invert=True, assume_unique=True)
                        cur_notin_pre = torch.isin(cache_eid, pre_bucket_eid_idx, invert=True, assume_unique=True)
                        pre_bucket_eid_idx[pre_notin_cur] = cache_eid[cur_notin_pre]
                        saveBin(pre_notin_cur.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_eid_incre_mask_{bucket_cache_start[-1]}.pt')
                        
                        cache_eid_incre = cache_eid[cur_notin_pre]
                    else:
                        pre_bucket_eid_idx = cache_eid

                bucket_cache_size += cache_nid_incre.shape[0] * 172 * 4 / 1024 ** 3
                bucket_cache_size += cache_eid_incre.shape[0] * 172 * 4 / 1024 ** 3 if (self.has_ef) else 0

                


                saveBin(pre_bucket_nid_idx.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/node_bucket_cache_idx_{bucket_cache_start[-1]}.pt')
                if (self.has_ef):
                    saveBin(pre_bucket_eid_idx.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/edge_bucket_cache_idx_{bucket_cache_start[-1]}.pt')
                

                for cur_nid_uni, cur_eid_uni,cur_batch_num in bucket_his:
                    cur_eid = cur_eid_uni
                    if (self.has_ef):
                        if (pre_eid is not None):
                            posincache = torch.isin(cur_eid_uni, pre_bucket_eid_idx, assume_unique=True)
                            posinpre = torch.isin(cur_eid_uni, pre_eid, assume_unique=True)
                            pos_cache_indices = find_indices(cur_eid_uni[posincache], pre_bucket_eid_idx.to(torch.int32))

                            cur_eid = cur_eid_uni[torch.bitwise_and(~posincache, ~posinpre)]

                            saveBin(posincache.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_pos_in_cache.pt')
                            saveBin(posinpre.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_pos_in_pre.pt')
                            saveBin(pos_cache_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_pos_cache_indices.pt')
                            

                        if (self.edge_feat_dim > 0):
                            edge_his.append(cur_eid.cpu())
                            edge_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_feat.bin')
                            edge_his_max.append(torch.max(cur_eid).cpu() if cur_eid.shape[0] > 0 else 0)
                            his_mem_byte += cur_eid.numel() * cur_eid.element_size()

                    cur_nid = cur_nid_uni
                    if (pre_nid is not None):
                        posincache = torch.isin(cur_nid_uni, pre_bucket_nid_idx, assume_unique=True)
                        posinpre = torch.isin(cur_nid_uni, pre_nid, assume_unique=True)
                        pos_cache_indices = find_indices(cur_nid_uni[posincache], pre_bucket_nid_idx.to(torch.int32))
                        cur_nid = cur_nid_uni[torch.bitwise_and(~posincache, ~posinpre)]

                        saveBin(posincache.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_pos_in_cache.pt')
                        saveBin(posinpre.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_pos_in_pre.pt')
                        saveBin(pos_cache_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_pos_cache_indices.pt')


                    if (self.node_feat_dim > 0):
                        node_his.append(cur_nid.cpu())
                        node_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_feat.bin')
                        node_his_max.append(torch.max(cur_nid).cpu() if cur_nid.shape[0] > 0 else 0)
                        his_mem_byte += cur_nid.numel() * cur_nid.element_size()

                    pre_eid = cur_eid_uni
                    pre_nid = cur_nid_uni
                if (self.has_ef):
                    edge_his.append(cache_eid_incre.cpu())
                    edge_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_edge_feat_{bucket_cache_start[-1]}.pt')
                    edge_his_max.append(torch.max(cache_eid_incre).cpu() if cache_eid_incre.shape[0] > 0 else 0)
                    his_mem_byte += cache_eid_incre.numel() * cache_eid_incre.element_size()

                node_his.append(cache_nid_incre.cpu())
                node_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_node_feat_{bucket_cache_start[-1]}.pt')
                node_his_max.append(torch.max(cache_nid_incre).cpu() if cache_nid_incre.shape[0] > 0 else 0)
                his_mem_byte += cache_nid_incre.numel() * cache_nid_incre.element_size()

                bucket_cache_start.append(left // batch_size)
                bucket_his = []
                bucket_eid_count.fill_(0)
                bucket_nid_count.fill_(0)
            
            bucket_his.append([nid_uni, eid_uni, batch_num])
            bucket_nid_count[nid_uni.long()] += 1
            # bucket_eid_count[eid_uni.long()] += 1
            


            
            if (his_mem_byte >= his_mem_threshold):
                
                self.stream_extract(node_feat_path, node_save_paths,node_window_size,node_his,node_his_max,self.node_feat_dim,np.float32)
                node_his.clear()
                node_his_max.clear()
                if (self.has_ef):
                    if (edge_window_size > 0):
                        self.stream_extract(edge_feat_path, edge_save_paths,edge_window_size,edge_his,edge_his_max,self.edge_feat_dim,np.float32)
                    edge_his.clear()
                    edge_his_max.clear()
                # his_ind.clear()
                his_mem_byte = 0
            
            
            print(f"total node num:{nid_uni.shape[0]},total edge num:{eid_uni.shape[0]},{left}--{right},max_eid:{torch.max(eid_uni)}, {his_mem_byte / 1024 ** 2:.0f}MB:{his_mem_threshold / 1024 **2:.0f}MB batch: {batch_num} batchsize: {batch_size} use time:{time.time() - start:.7f}s")
            
            left = right
            batch_num += 1
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        bucket_nid_count_sort,bucket_nid_count_sort_idx = torch.sort(bucket_nid_count, descending=True)
        cache_nid = bucket_nid_count_sort_idx[:node_idx_budget]
        cache_nid_incre = cache_nid
        if(self.d == "GDELT"):
            if (self.has_ef):
                bucket_eid_count_cpu = bucket_eid_count.cpu()
                
                _, bucket_eid_count_sort_idx_cpu = torch.sort(bucket_eid_count_cpu, descending=True)

                bucket_eid_count_sort_idx = bucket_eid_count_sort_idx_cpu.cuda()
                cache_eid = bucket_eid_count_sort_idx[:edge_idx_budget]
                cache_eid_incre = cache_eid
        else:
            if (self.has_ef):
                _,bucket_eid_count_sort_idx = torch.sort(bucket_eid_count, descending= True)
                cache_eid = bucket_eid_count_sort_idx[:edge_idx_budget]
                cache_eid_incre = cache_eid
        gc.collect()
        torch.cuda.empty_cache()
        
        
        
        if (pre_bucket_nid_idx is not None):

            pre_notin_cur = torch.isin(pre_bucket_nid_idx, cache_nid, invert=True, assume_unique=True)
            cur_notin_pre = torch.isin(cache_nid, pre_bucket_nid_idx, invert=True, assume_unique=True)
            pre_bucket_nid_idx[pre_notin_cur] = cache_nid[cur_notin_pre]
            saveBin(pre_notin_cur.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_nid_incre_mask_{bucket_cache_start[-1]}.pt')
            cache_nid_incre = cache_nid[cur_notin_pre]

        else:
            pre_bucket_nid_idx = cache_nid

        if (self.has_ef):
            if (pre_bucket_eid_idx is not None):
                pre_notin_cur = torch.isin(pre_bucket_eid_idx, cache_eid, invert=True, assume_unique=True)
                cur_notin_pre = torch.isin(cache_eid, pre_bucket_eid_idx, invert=True, assume_unique=True)
                pre_bucket_eid_idx[pre_notin_cur] = cache_eid[cur_notin_pre]
                saveBin(pre_notin_cur.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_eid_incre_mask_{bucket_cache_start[-1]}.pt')
                
                cache_eid_incre = cache_eid[cur_notin_pre]
            else:
                pre_bucket_eid_idx = cache_eid

        bucket_cache_size += cache_nid_incre.shape[0] * 172 * 4 / 1024 ** 3
        bucket_cache_size += cache_eid_incre.shape[0] * 172 * 4 / 1024 ** 3 if (self.has_ef) else 0

        


        saveBin(pre_bucket_nid_idx.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/node_bucket_cache_idx_{bucket_cache_start[-1]}.pt')
        if (self.has_ef):
            saveBin(pre_bucket_eid_idx.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/edge_bucket_cache_idx_{bucket_cache_start[-1]}.pt')
        

        for cur_nid_uni, cur_eid_uni,cur_batch_num in bucket_his:
            cur_eid = cur_eid_uni
            if (self.has_ef):
                if (pre_eid is not None):
                    posincache = torch.isin(cur_eid_uni, pre_bucket_eid_idx, assume_unique=True)
                    posinpre = torch.isin(cur_eid_uni, pre_eid, assume_unique=True)
                    pos_cache_indices = find_indices(cur_eid_uni[posincache], pre_bucket_eid_idx.to(torch.int32))

                    cur_eid = cur_eid_uni[torch.bitwise_and(~posincache, ~posinpre)]

                    saveBin(posincache.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_pos_in_cache.pt')
                    saveBin(posinpre.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_pos_in_pre.pt')
                    saveBin(pos_cache_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_pos_cache_indices.pt')
                    
                if (self.edge_feat_dim > 0):
                    edge_his.append(cur_eid.cpu())
                    edge_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_edge_feat.bin')
                    # edge_save_paths.append()
                    edge_his_max.append(torch.max(cur_eid).cpu() if cur_eid.shape[0] > 0 else 0)
                    his_mem_byte += cur_eid.numel() * cur_eid.element_size()

            cur_nid = cur_nid_uni
            if (pre_nid is not None):
                posincache = torch.isin(cur_nid_uni, pre_bucket_nid_idx, assume_unique=True)
                posinpre = torch.isin(cur_nid_uni, pre_nid, assume_unique=True)
                pos_cache_indices = find_indices(cur_nid_uni[posincache], pre_bucket_nid_idx.to(torch.int32))
                # self.bucket_cache_node_feat[node_cache_indices]
                cur_nid = cur_nid_uni[torch.bitwise_and(~posincache, ~posinpre)]

                saveBin(posincache.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_pos_in_cache.pt')
                saveBin(posinpre.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_pos_in_pre.pt')
                saveBin(pos_cache_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_pos_cache_indices.pt')


            if (self.node_feat_dim > 0):
                node_his.append(cur_nid.cpu())
                node_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{cur_batch_num}_node_feat.bin')
                node_his_max.append(torch.max(cur_nid).cpu() if cur_nid.shape[0] > 0 else 0)
                his_mem_byte += cur_nid.numel() * cur_nid.element_size()

            pre_eid = cur_eid_uni
            pre_nid = cur_nid_uni
        if (self.has_ef):
            edge_his.append(cache_eid_incre.cpu())
            edge_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_edge_feat_{bucket_cache_start[-1]}.pt')
            edge_his_max.append(torch.max(cache_eid_incre).cpu() if cache_eid_incre.shape[0] > 0 else 0)
            his_mem_byte += cache_eid_incre.numel() * cache_eid_incre.element_size()

        node_his.append(cache_nid_incre.cpu())
        node_save_paths.append(path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/cache_node_feat_{bucket_cache_start[-1]}.pt')
        node_his_max.append(torch.max(cache_nid_incre).cpu() if cache_nid_incre.shape[0] > 0 else 0)
        his_mem_byte += cache_nid_incre.numel() * cache_nid_incre.element_size()

        bucket_cache_start.append(left // batch_size)
        bucket_his = []
        bucket_eid_count.fill_(0)
        bucket_nid_count.fill_(0)
    

            

        self.stream_extract(node_feat_path, node_save_paths,node_window_size,node_his,node_his_max,self.node_feat_dim,np.float32)
        if (edge_window_size > 0 and self.has_ef):
            self.stream_extract(edge_feat_path, edge_save_paths,edge_window_size,edge_his,edge_his_max,self.edge_feat_dim,np.float32)

        bucket_config_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/' + f'/bucket_config.json'
        with open(bucket_config_path, 'w') as json_file:
            json.dump(res_config, json_file, indent=4)



    def gen_part_stream_bucket_cache(self, budget = (10 * 1024), bucket_budget = 1 * 1024 ** 3):
        d = self.d
        path = self.path


        df = self.df
        batch_size = self.batch_size

        budget_byte = budget * 1024 * 1024
        his_ind = []
        node_his = []
        edge_his = []
        node_his_max = []
        edge_his_max = []
        node_window_size = int(budget_byte * 0.9 / 4 / self.node_feat_dim)
        edge_window_size = int(budget_byte * 0.9 / 4 / self.edge_feat_dim) if self.edge_feat_dim != 0 else 0
        history_datas = []
        his_mem_threshold = 0.05 * 1024 ** 3  # 4GB
        his_mem_byte = 0

        res_config = {}
        res_config['bucket_ptr'] = [0]
        res_config['max_node_num'] = 0
        res_config['max_edge_num'] = 0

        left, right = 0, 0
        batch_num = 0
        datas = self.datas
        pre_eid, pre_nid = None, None
        edge_end = datas['src'].shape[0]
        if (self.d == 'MAG'):
            total_src = datas['src'].pin_memory().to(torch.int32)
            total_dst = datas['dst'].pin_memory().to(torch.int32)
            total_time = datas['time'].pin_memory().to(torch.float32)
            total_eid = datas['eid'].pin_memory().to(torch.int32)
        else:
            total_src = datas['src'].cuda().to(torch.int32)
            total_dst = datas['dst'].cuda().to(torch.int32)
            total_time = datas['time'].cuda().to(torch.float32)
            total_eid = datas['eid'].cuda().to(torch.int32)
        
        count = torch.zeros(121751665, dtype = torch.int32, device = 'cuda:0')
        batch_count = []
        # eid_uni = torch.
        while True:
            start = time.time()
            first_flag = True

            right += batch_size
            right = min(edge_end, right)
            if (left >= right):
                break
            first_flag = False

            src = total_src[left: right]
            dst = total_dst[left: right]
            times = total_time[left: right]
            eid = total_eid[left: right]
            root_nodes = torch.cat((src, dst)).cuda()
            root_ts = torch.cat((times, times)).cuda()

            ret_list = self.sampler.sample_layer(root_nodes, root_ts)
            nid_uni = torch.unique(root_nodes).to(torch.int32).cuda()

            for ret in ret_list:
                src,dst,outts,outeid,root_nodes,root_ts,dts = ret
                eid = outeid[outeid > -1]

                cur_eid = torch.unique(eid)

            ret = ret_list[-1]
            src,dst,outts,outeid,root_nodes,root_ts,dts = ret
            del ret_list
            del outts, outeid, root_ts, dst
            # emptyCache()
            mask = src > -1
            src = src[mask]
            nid_uni = torch.cat((src, root_nodes))
            nid_uni = torch.unique(nid_uni)
            nid_uni,_ = torch.sort(nid_uni)
            # eid_uni,_ = torch.sort(eid_uni)

            count[nid_uni.long()] += 1
            batch_count.append(nid_uni)


            res_config['max_node_num'] = int(max(res_config['max_node_num'], nid_uni.shape[0] * 1.5))
            if (first_flag):
                break
            his_ind.append(batch_num)
            res_config['bucket_ptr'].append(right)


            threshold = 50
            cache_num = 10000000
            if (batch_num > 0 and batch_num % threshold == 0):
                count_sort, count_idx = torch.sort(count, descending=True)
                cache_idx = count_idx[:cache_num]
                cur_count = 0
                for batch_info in batch_count:
                    nid_uni = batch_info
                    cur_nid = nid_uni
                    
                    if (pre_nid is not None):
                        nid_incre_mask = torch.isin(nid_uni, cache_idx, assume_unique=True, invert=True)
                        nid_incre_mask = torch.bitwise_and(nid_incre_mask, torch.isin(nid_uni, pre_nid, assume_unique=True, invert = True))
                        
                        cur_nid = nid_uni[nid_incre_mask]
                        saveBin(nid_incre_mask.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num-threshold+cur_count}_node_map_incre_mask.pt')

                    if (self.node_feat_dim > 0):
                        node_his.append(cur_nid.cpu())
                        node_his_max.append(torch.max(cur_nid).cpu())
                        his_mem_byte += cur_nid.numel() * cur_nid.element_size()

                    saveBin(nid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num-threshold+cur_count}_node_map.pt')
                    pre_nid = nid_uni

                    cur_count += 1

                count.fill_(0)
                batch_count = []
                saveBin(cache_idx.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/bucket_cache_{batch_num-threshold}_nidx.bin')
                    


            
            if (his_mem_byte >= his_mem_threshold):
                node_feat_path = f'{path}/node_features.bin'
                edge_feat_path =  f'{path}/edge_features.bin'
                node_save_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/parti_node_feat.bin'
                edge_save_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/parti_edge_feat.bin'
                self.stream_extract(node_feat_path, node_save_path,node_window_size,node_his,his_ind,node_his_max,self.node_feat_dim,getattr(np, self.node_feat_type))
                node_his.clear()
                node_his_max.clear()


                his_ind.clear()
                his_mem_byte = 0
            sampleTime = time.time() - start
            
            print(f"total node num:{nid_uni.shape[0]}, {his_mem_byte / 1024 ** 2:.0f}MB:{his_mem_threshold / 1024 **2:.0f}MB batch: {batch_num} batchsize: {batch_size} use time:{time.time() - start:.7f}s")
            
            left = right
            batch_num += 1
        # batch_num -= 1
        if (batch_count):
            pre_batch_count = len(batch_count)
            count_sort, count_idx = torch.sort(count, descending=True)
            cache_idx = count_idx[:cache_num]

            for i, batch_info in enumerate(batch_count):
                
                nid_uni = batch_info
                cur_nid = nid_uni
                if (pre_nid is not None):
                    nid_incre_mask = torch.isin(nid_uni, pre_nid, assume_unique=True, invert = True)
                    nid_incre_mask = torch.bitwise_and(nid_incre_mask, torch.isin(nid_uni, cache_idx, assume_unique=True, invert=True))
                    cur_nid = nid_uni[nid_incre_mask]
                    saveBin(nid_incre_mask.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num - pre_batch_count + i}_node_map_incre_mask.pt')

                if (cur_nid.shape[0] > 0 and self.node_feat_dim > 0):
                    node_his.append(cur_nid.cpu())
                    node_his_max.append(torch.max(cur_nid).cpu())
                    his_mem_byte += cur_nid.numel() * cur_nid.element_size()

                saveBin(nid_uni.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/part{batch_num - pre_batch_count + i}_node_map.pt')
                pre_nid = nid_uni

            count.fill_(0)
            batch_count = []
            saveBin(cache_idx.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/bucket_cache_{batch_num - pre_batch_count}_nidx.bin')
            
        node_feat_path = f'{path}/node_features.bin'
        edge_feat_path =  f'{path}/edge_features.bin'
        node_save_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/parti_node_feat.bin'
        edge_save_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/parti_edge_feat.bin'
        self.stream_extract(node_feat_path, node_save_path,node_window_size,node_his,his_ind,node_his_max,self.node_feat_dim,getattr(np, self.node_feat_type))
        node_his.clear()

        if (edge_window_size > 0):
            self.stream_extract(edge_feat_path, edge_save_path,edge_window_size,edge_his,his_ind,edge_his_max,self.edge_feat_dim,getattr(np, self.edge_feat_type))
        edge_his.clear()
        node_his_max.clear()
        edge_his_max.clear()
        his_ind.clear()

        bucket_config_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}/' + f'/bucket_config.json'
        with open(bucket_config_path, 'w') as json_file:
            json.dump(res_config, json_file, indent=4)

    def incre_strategy(self, pre_map, cur_id):
        pre_map_sort, pre_map_sort_indices = torch.sort(pre_map)
        cur_id_sort, cur_id_sort_indices = torch.sort(cur_id)
        table1 = torch.zeros_like(pre_map_sort, dtype = torch.int32, device = 'cuda:0') - 1
        table2 = torch.zeros_like(cur_id_sort, dtype = torch.int32, device = 'cuda:0') - 1
        
        dgl.findSameIndex(pre_map_sort, cur_id_sort, table1, table2)

        incre_map_index = pre_map_sort_indices[table1==-1] 
        incre_nid = cur_id[cur_id_sort_indices[table2==-1].long()] 
        incre_nfeat = torch.randn((incre_nid.shape[0],10), dtype = torch.float32)
        pre_map[incre_map_index] = -1

        indices_able = torch.nonzero(pre_map == -1).reshape(-1)
        len_able = indices_able.shape[0]
        if (len_able < incre_nid.shape[0]):
            pre_map = torch.cat((pre_map, torch.zeros(incre_nid.shape[0] - len_able, dtype = torch.int32, device = 'cuda:0') - 1))
            indices_able = torch.nonzero(pre_map == -1).reshape(-1)

        incre_indices = indices_able[:incre_nid.shape[0]]
        pre_map[incre_indices] = incre_nid

        return pre_map, incre_indices, incre_nid


    def gen_part_incre(self):
        d = self.d
        path = self.path

        df = self.df
        batch_size = self.batch_size
        train_edge_end = self.train_edge_end
        group_indexes = np.array(df[:train_edge_end].index // batch_size)

        edge_map, node_map = None,None
        max_edge_num, max_node_num = 0, 0

        for batch_num, rows in df[:train_edge_end].groupby(group_indexes):
            emptyCache()
            root_nodes = torch.from_numpy(np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)).cuda()
            root_ts = torch.from_numpy(np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)).cuda()


            start = time.time()
            ret_list = self.sampler.sample_layer(root_nodes, root_ts)
            eid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')
            nid_uni = torch.empty(0, dtype = torch.int32, device = 'cuda:0')

            for ret in ret_list:
                src,dst,outts,outeid,root_nodes,root_ts,dts = ret
                eid = outeid[outeid > -1]

                cur_eid = torch.unique(eid)
                eid_uni = torch.cat((cur_eid, eid_uni))
                eid_uni = torch.unique(eid_uni)
            
            ret = ret_list[-1]
            src,dst,outts,outeid,root_nodes,root_ts,dts = ret
            del ret_list
            del outts, outeid, root_ts, dst
            emptyCache()

            mask = src > -1
            src = src[mask]
            nid_uni = torch.cat((src, root_nodes))
            nid_uni = torch.unique(nid_uni)

            eid_uni,_ = torch.sort(eid_uni)
            if (edge_map is None):
                edge_map = torch.tensor([2**31 - 1], dtype = torch.int32, device = 'cuda:0')
            cur_edge_num = eid_uni.shape[0]
            max_edge_num = max(cur_edge_num, max_edge_num)
            edge_map, incre_edge_indices, incre_eid = self.incre_strategy(edge_map, eid_uni)
            cur_edge_feat = self.select_index('edge_feats',incre_eid.to(torch.int64))
            saveBin(cur_edge_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}-incre/part{batch_num}_incre_edge_feat.pt')
            saveBin(edge_map.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}-incre/part{batch_num}_incre_edge_map.pt')
            saveBin(incre_edge_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}-incre/part{batch_num}_incre_edge_indices.pt')

            nid_uni,_ = torch.sort(nid_uni)
            if (node_map is None):
                node_map = torch.tensor([2**31 - 1], dtype = torch.int32, device = 'cuda:0')
            
            cur_node_num = eid_uni.shape[0]
            max_node_num = max(cur_node_num, max_node_num)
            node_map, incre_node_indices, incre_nid = self.incre_strategy(node_map, nid_uni)
            cur_node_feat = self.select_index('node_feats',incre_nid.to(torch.int64))
            saveBin(cur_node_feat.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}-incre/part{batch_num}_incre_node_feat.pt')
            saveBin(node_map.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}-incre/part{batch_num}_incre_node_map.pt')
            saveBin(incre_node_indices.cpu(), path + f'/part-{self.batch_size}-{self.sampler.fan_nums}-incre/part{batch_num}_incre_node_indices.pt')

            sampleTime = time.time() - start
            
            del root_nodes,eid_uni,nid_uni,src,mask,eid,_

        import json
        conf_path = path + f'/part-{self.batch_size}-{self.sampler.fan_nums}-incre/part-info.json'
        data = {'max_edge_num': max_edge_num, 'max_node_num': max_node_num}
        with open(conf_path, 'w') as f:
            json.dump(data, f)
