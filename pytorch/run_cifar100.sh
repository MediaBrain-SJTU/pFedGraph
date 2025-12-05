data_partition="noniid-skew"
client=10
iternum=400
skew_class=20
alpha=0.8
beta=0.1

# Baselines
python local_training.py --gpu "7" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedavg.py --gpu "7" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedavg_ft.py --gpu "7" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedprox.py --gpu "5" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedprox_ft.py --gpu "5" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python cfl.py --gpu "4" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python perfedavg.py --gpu "5" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python pfedme.py --gpu "7" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedamp.py --gpu "5" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedrep.py --gpu "5" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_rep_iterations $iternum --beta $beta
python ditto.py --gpu "4" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python pfedhn.py --gpu "7" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedrod.py --gpu "7" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python knn_per.py --gpu "7" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta

# Ours
python pfedgraph_cosine.py --gpu "4" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "all"
python pfedgraph_approx.py --gpu "4" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "all"
python pfedgraph_cosine.py --gpu "5" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "fc"
python pfedgraph_approx.py --gpu "5" --dataset cifar100 --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "fc"