import argparse
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='LASTFM')
parser.add_argument('--config', type=str, help='path to config file', default='/DOLPHIN/config/TGN-1.yml')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--no_emb_buffer', action='store_true', default=True)

parser.add_argument('--only_gen_part', action='store_true', default=False)
parser.add_argument('--use_async_prefetch', action='store_true', default=False)
parser.add_argument('--use_stream', action='store_true', default=True)
parser.add_argument('--use_disk', action='store_true', default=True)
parser.add_argument('--dis_threshold', type=int, default=10, help='distance threshold')
parser.add_argument('--fanout', type=int, default=-1)
parser.add_argument('--rand_edge_features', type=int, default=128, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=128, help='use random node featrues')
parser.add_argument('--substream_size', type=int, default=60000, help='substream size')
parser.add_argument('--cache_budget', type=int, default=10)
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
parser.add_argument('--opt', action='store_true', default=False)
args=parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
import sys
import os
total_start = time.time()
root_dir = '/DOLPHIN'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config.train_conf import *
GlobalConfig.conf = 'basic_conf_disk.json'
use_dynamic_k = True


from modules import *
from sampler.sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import emptyCache
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == '__main__':

    node_feats, edge_feats = None,None
    if (not args.use_stream and not args.use_disk):
        node_feats, edge_feats = load_feat(args.data)

    if (not args.use_stream):
        g, df = load_graph(args.data)
        print("g['indices'].shape[0]",g['indices'].shape[0])
        # train_edge_end = 0
        if (args.data in ['BITCOIN']):
            train_edge_end = 86063713
            val_edge_end = 110653345
        else:
            train_edge_end = df[df['ext_roll'].gt(0)].index[0]
            val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    else:
        g, datas, df_conf = load_graph_bin(args.data)
        if (args.data in ['BITCOIN']):
            train_edge_end = 86063713
            val_edge_end = 110653345
        else:
            train_edge_end = df_conf['train_edge_end']
            val_edge_end = df_conf['val_edge_end']
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    if (args.fanout != -1):
        sample_param['neighbor'] = [args.fanout]
        
    if (args.data == 'GDELT' and sample_param['layer'] == 2):
        sample_param['neighbor'] = [8, 8]
        train_param['epoch'] = 1
    # gnn_dim_node = 0 if (node_feats.shape[0] == 0) else node_feats.shape[1]
    # gnn_dim_edge = 0 if (edge_feats is None or edge_feats.shape[0] == 0) else edge_feats.shape[1]
    if (args.data == 'LASTFM'):
        gnn_dim_edge = 128
        gnn_dim_node = 128
    elif (args.data == 'LASTFM1'):
        gnn_dim_edge = 128
        gnn_dim_node = 128   
    elif (args.data == 'TALK'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (args.data == 'STACK'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (args.data == 'GDELT'):
        gnn_dim_edge = 182
        gnn_dim_node = 413
    elif (args.data == 'BITCOIN'):
        gnn_dim_edge = 172
        gnn_dim_node = 172
    elif (args.data == 'MAG'):
        gnn_dim_edge = 0
        gnn_dim_node = 768
    elif (args.data == 'MOOC'):
        gnn_dim_edge = 0
        gnn_dim_node = 172
    elif (args.data == 'LASTFM_interval'):
        gnn_dim_edge = 128
        gnn_dim_node = 128
    else:
        raise RuntimeError("have not this dataset config!")
    


    

    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True


    from sampler.sampler_gpu import *
    use_gpu_sample = False
    use_gpu_sample = True
    no_neg = 'no_neg' in sample_param and sample_param['no_neg']

    
    
    sampler_gpu = Sampler_GPU(g, sample_param['neighbor'], sample_param['layer'], None)

    node_num = g['indptr'].shape[0] - 1
    edge_num = g['indices'].shape[0]

    
    
    if (args.data == 'LASTFM'):
        edge_num = 1293103

    elif (args.data == 'TALK'):
        edge_num = 7833140
    elif (args.data == 'STACK'):
        edge_num = 63497048  
    elif (args.data == 'BITCOIN'):
        edge_num = 122948162 
    elif (args.data == 'LASTFM_interval'):
        edge_num = 1293103    
    elif (args.data == 'MAG'):
        edge_num =  1297748926 
    else:
        raise RuntimeError("have not this dataset config!")
    
    cur_train = {
        'node_feat_type': 'float32',
        'edge_feat_type': 'float32',
        'dataset': args.data,
        'node_feat_dim': gnn_dim_node,
        'edge_feat_dim': gnn_dim_edge,
        'node_num': node_num,
        'edge_num': edge_num
    }
    with open('/DOLPHIN/cur_train.json', mode = 'w') as f:
        json.dump(cur_train, f)
    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method("spawn")
    from pre_fetch import *
    use_async_prefetch = args.use_async_prefetch

    parent_conn = None
    prefetch_conn = None
    if (use_async_prefetch):
        parent_conn, child_conn = multiprocessing.Pipe()
        prefetch_conn, prefetch_child_conn = multiprocessing.Pipe()

        p = multiprocessing.Process(target=prefetch_worker, args=(child_conn, prefetch_child_conn))
        p.start()

        parent_conn.send(('init_feats', (args.data, )))
        print(f"Sent: {'init_feats'}")
        result = parent_conn.recv()
        print(f"Received: {result}")
        node_feats,edge_feats = 1,1

    prefetch_only_conn = prefetch_conn
    prefetch_conn = parent_conn
    

    sampler = None
    # if (True):
    if not (('no_sample' in sample_param and sample_param['no_sample']) or (use_gpu_sample)):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                sample_param['strategy']=='recent', sample_param['prop_time'],
                                sample_param['history'], float(sample_param['duration']))





    from feat_buffer import *
    # gpu_sampler = Sampler_GPU(g, 10)

    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1, g['indptr'][-1])
    use_val_test = True
    if (not args.use_stream):
    
        feat_buffer = Feat_buffer(args.data, df, None, train_param, memory_param, train_edge_end, args.substream_size//train_param['batch_size'],args.substream_size,\
                                sampler_gpu,neg_link_sampler, prefetch_conn=(prefetch_conn, prefetch_only_conn), feat_dim = (gnn_dim_node, gnn_dim_edge))
        feat_buffer.init_feat(node_feats, edge_feats)
        
        feat_buffer.gen_part(incre = True, mode = '', save_feat= False)
        feat_buffer.val_edge_end = val_edge_end
        feat_buffer.test_edge_end = len(df)
    
    else:
       
        feat_buffer = Feat_buffer(args.data, None, datas, train_param, memory_param, train_edge_end, args.substream_size//train_param['batch_size'],args.substream_size,sampler_gpu,neg_link_sampler, node_num = node_num, edge_num=edge_num, prefetch_conn=(prefetch_conn, prefetch_only_conn), feat_dim = (gnn_dim_node, gnn_dim_edge), substream_size=args.substream_size)
        feat_buffer.train_edge_end = train_edge_end
        if (not use_async_prefetch):
            feat_buffer.init_feat(node_feats, edge_feats)
        
        if (args.data == 'MAG1' or args.data == 'MOOC1'):
            feat_buffer.gen_part_stream_bucket_cache()
        else:
            
            feat_buffer.gen_part_stream(bucket_optimal=False, cache_budget=args.cache_budget * 1024 ** 3)
            
    print_total()
    flush_saveBin_conf()
    print(f"disk preprocess total use time {time.time() - total_start:.4f}s")