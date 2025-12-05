data_partition="noniid-skew"
client=10
iternum=200
alpha=0.8
beta=0.1

# Baselines
# python fedavg.py --gpu "7" --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta

# Ours
time python pfedgraph_cosine.py --gpu "1" --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "fc"