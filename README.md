# LazyGCN

# Installation

`pip install requirement.txt`

`python setup.py develop`

# Usage

`import lazygcn`



## Loading dataset

set dataset directory

`os.environ["GNN_DATASET_DIR"] = "<your dataset directory>"`

`dataset = data.Dataset('pubmed')`

Supported datasets: 
- [Cora](https://data.dgl.ai/dataset/cora_raw.zip)
- [Citeseer](https://data.dgl.ai/dataset/citeseer.zip)
- [Pubmed](https://data.dgl.ai/dataset/pubmed.zip)
- [Flickr, PPI, PPI-Large, Reddit, Yelp and Amazon2M](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz)

## Running a model

There are several predefined training class, which prepare data and create pre-defined models and optimizer.

- Models: GCN, MLP, RecGCN [GraphSAGE, GAT]*
- Layers: GConv, Linear, RecGConv
- Minibatch: Full, Random, Split
- NodeSampler: LayerWise, LADIES, Exact, FullGraph [FastGCN, NodeWise]*



See `notebooks` folder for examples and `lazygcn/training` for pre-defined trainer.