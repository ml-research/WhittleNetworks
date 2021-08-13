# Whittle Networks: A Deep Likelihood Model for Time Series

This repository is the official implementation of Whittle Networks introduced in 
[Whittle Networks: A Deep Likelihood Model for Time Series](https://ml-research.github.io/papers/yu2021icml_wspn.pdf) by Zhongjie Yu, Fabrizio Ventola, and Kristian Kersting, published at ICML 2021.

## Setup

This will clone the repo, install a Python virtual env (requires Python 3.6), and the required packages.

    git clone https://github.com/ml-research/WhittleNetworks.git
    ./setup.sh

## Datasets

Download datasets from [TU Datalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2887/), and unzip:

    wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2887/data.zip
    unzip data.zip

## Demos

### Activate the virtual environment:

    source ./venv_wnet/bin/activate

### To train WSPN on some of the datasets:

    ./run_WSPN.sh Sine

"Sine" can be replaced with "MNIST", "SP", "Stock", or "Billiards".

This will train and evaluate WSPNs with 1d, pair, and 2d Gaussian leaf nodes. Details can be found in Table 1 in our paper.

### To extract the conditional independence structure from a WSPN:

    python script_graph.py --data_type=Sine --graph_type=bn

"Sine" can be replaced with "SP", "Stock", or "VAR".

```--graph_type``` can be either "bn" -- directed graph, or "mn" -- undirected graph.

Bayesian information criterion will be enabled with ```--BIC```

Pre-trained WSPN models are in ```results/```

### To train and evaluate conditional WSPN for forecasting on the Mackey-Glass dataset: 

    python script_wcspn.py

### To train and evaluate Whittle AE:

    python train_WhittleAE.py
    python test_WhittleAE.py


## Citation
If you find this code useful in your research, please consider citing:


    @inproceedings{yu2021wspn,
      title = {Whittle Networks: A Deep Likelihood Model for Time Series}, 
      author = {Yu, Zhongjie and Ventola, Fabrizio and Kersting, Kristian}, 
      booktitle = { Proceedings of the International Conference on Machine Learning (ICML) },
      pages = {12177--12186},
      year = {2021}
    } 

## Acknowledgments

* This work is supported by the Federal Ministry of Education and Research (BMBF; project "MADESI", FKZ 01IS18043B, and Competence Center for AI and Labour; "kompAKI", FKZ 02L19C150), the German Science Foundation (DFG, German Research Foundation; GRK 1994/1 "AIPHES"), the Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK; projects "The Third Wave of AI" and "The Adaptive Mind"), and the Hessian research priority programme LOEWE within the project "WhiteBox".

