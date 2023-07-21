# Personalized Federated Learning with Inferred Collaboration Graphs, ICML2023

**Note:** This repo is still in progress.

## Dependencies
* PyTorch = 1.12.0

## Quick Start

- Run FedAvg algorithm on CIFAR-10, skew partition, 10 clients, local iteration number is 200:

```console
python fedavg.py --gpu "7" --dataset 'cifar10' --partition 'noniid-skew' --n_parties 10 --num_local_iterations 200
```

- For fair comparisons on CIFAR-10, you can run the bash file. (Same to CIFAR-100 and Yahoo! Answers)

```console
sh run_cifar10.sh
```

## pFedGraph
- pfedgraph_approx.py runs the experiment that we regularize the local model with the unnormalized aggregated model.
- pfedgraph_cosine.py runs the experiment that we regularize the local model with the normalized aggregated model.

## Others

- Some of the codes are adapted from https://github.com/TsingZ0/PFL-Non-IID

## Cite

To cite, please use:

```latex
@inproceedings{ye2023pretraining,
  title={Personalized Federated Learning with Inferred Collaboration Graphs},
  author={Ye, Rui and Ni, Zhenyang and Wu, Fangzhao and Chen, Siheng and Wang, Yanfeng},
  booktitle={International Conference on Machine Learning},
  pages={39801--39817},
  year={2023},
  organization={PMLR}
}
```