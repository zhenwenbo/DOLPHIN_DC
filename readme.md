# DOLPHIN
Code repository for the SC 2026 under review paper:
"DOLPHIN: Scalable Disk–RAM–GPU Pipelined Training for Massive Temporal GNNs"

```

/
├── config/
├── preprocess/
│   ├── disk_preprocess.py -- for DOLPHIN Preprocessing (new)
│   ├── original_preprocess.py -- for converting original dataset to TGL format (only required for custom datasets) (new)
│   ├── generate_feature.py -- for generating random features for datasets without features (new)
│   ├── memory_preprocess.py -- for DOLPHIN Preprocessing
│   └── node_memory.py -- for DOLPHIN Preprocessing
├── sampler/
│   ├── build/
│   ├── sampler.py 
│   ├── sampler_gpu.py -- for GPU Sampler (new)
│   └── setup.py
├── dolphin-dgl/ -- for CUDA sampler compilation (new)
├── README.md
├── down.sh
├── feat_buffer.py 
├── gen_graph.py
├── layers.py
├── memorys.py
├── modules.py
├── pre_fetch.py --for Pipeline Optimization (new)
├── setup.py
├── train.py
└── utils.py


```
## Requirements
- python >= 3.8
- pytorch >= 1.12.1
- numpy >= 1.24.3
- dgl >= 0.9.1

Since DOLPHIN is based on the CUDA compilation process of the DGL framework, it is necessary to compile the given DGL version to obtain some CUDA functions used during DOLPHIN training.

```
cd dolphin-dgl
bash build.sh
```


## Prepare Datasets

Wiki-Talk and Stack-Overflow datasets need to be downloaded from http://snap.stanford.edu/data/wiki-talk-temporal.html and https://snap.stanford.edu/data/sx-stackoverflow.html respectively. The Bitcoin dataset can be downloaded from https://networkrepository.com/soc-bitcoin.php.

For efficient model testing and streamlined preparation, it is recommended to use the LASTFM dataset directly during testing—this dataset already comes in TGL format, eliminating the need for extra format conversion or feature generation, which significantly saves time for quick model validation (e.g., verifying function integrity or debugging code logic).

Five datasets involved in the paper (Wiki-Talk, Stack-Overflow, Bitcoin, GDELT, and MAG) lack built-in features, so they all require running `preprocess/generate_feature.py` to create synthetic features that match the model’s input requirements. 

Among them, Wiki-Talk, Stack-Overflow, and Bitcoin have raw data in non-TGL formats and need prior conversion using `preprocess/original_preprocess.py` to get TGL-format data, while GDELT (e.g., downloadable via down.sh) and MAG have raw data compatible with TGL, so no additional format conversion is needed.

For Wiki-Talk, Stack-Overflow, and Bitcoin, the standard workflow is to first download the raw data, then convert it to TGL format via `original_preprocess.py`, generate synthetic features via `generate_feature.py`, and finally use the processed data for model training or testing.

Possible Program Commands:
```
python /DOLPHIN/preprocess/original_preprocess.py --data dataset_name
python /DOLPHIN/preprocess/generate_feature.py
```

## Configuration
We provide some model configuration files in DOLPHIN/config, with specific configuration descriptions as follows (content sourced from the open-source framework TGL):
```
sampling:
  - layer: <number of layers to sample>
    neighbor: <a list of integers indicating how many neighbors are sampled in each layer>
    strategy: <'recent' that samples most recent neighbors or 'uniform' that uniformly samples neighbors form the past>
    prop_time: <False or True that specifies wherether to use the timestamp of the root nodes when sampling for their multi-hop neighbors>
    history: <number of snapshots to sample on>
    duration: <length in time of each snapshot, 0 for infinite length (used in non-snapshot-based methods)
    num_thread: <number of threads of the sampler>
memory: 
  - type: <'node', we only support node memory now>
    dim_time: <an integer, the dimension of the time embedding>
    deliver_to: <'self' that delivers the mails only to involved nodes or 'neighbors' that deliver the mails to neighbors>
    mail_combine: <'last' that use the latest latest mail as the input to the memory updater>
    memory_update: <'gru' or 'rnn'>
    mailbox_size: <an integer, the size of the mailbox for each node>
    combine_node_feature: <False or True that specifies whether to combine node features (with the updated memory) as the input to the GNN.
    dim_out: <an integer, the dimension of the output node memory>
gnn:
  - arch: <'transformer_attention' or 'identity' (no GNN)>
    layer: <an integer, number of layers>
    att_head: <an integer, number of attention heads>
    dim_time: <an integer, the dimension of the time embedding>
    dim_out: <an integer, the dimension of the output dynamic node embedding>
train:
  - epoch: <an integer, number of epochs to train>
    batch_size: <an integer, the batch size (of edges); for multi-gpu training, this is the local batchsize>
    reorder: <(optional) an integer that is divisible by batch size the specifies how many chunks per batch used in the random chunk scheduling>
    lr: <floating point, learning rate>
    dropout: <floating point, dropout>
    att_dropout: <floating point, dropout for attention>
    all_on_gpu: <False or True that decides if the node/edge features and node memory are completely stored on GPU>
```
Corresponding to the paper, we provide initial configuration files for the TGAT, TGN, and TimeSGN models under 1-layer and 2-layer message passing in DOLPHIN/config.

## Preprocessing
It is necessary to complete the dataset preparation and model configuration according to the Prepare Datasets step and the file configuration step beforehand. Execute:

```
python /DOLPHIN/preprocess/memory_preprocess.py --data dataset_name
python /DOLPHIN/preprocess/disk_preprocess.py --data dataset_name
python /DOLPHIN/preprocess/node_memory.py --data dataset_name
python /DOLPHIN/eid_analyze/eid_analyze_parallel.py 
```

Please note that for different models, preprocessing only needs to be performed once under the same sampling configuration. However, different sampling parameters (such as different layer numbers or different fanouts) require re-execution of the preprocessing operation.

## Train

After completing the preprocessing operation, you can run the training program by specifying the dataset name and model configuration file path:
```
python /DOLPHIN/train.py --data dataset_name  --eid_load final --config model_config_file_path
```
