data_partition="iid"
client=10
iternum=200
skew_class=2
alpha=0.8
beta=0.1
dataset="yahoo_answers"
datadir='./data'

# Baselines
python local_training.py --gpu "5" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedavg_ft.py --gpu "5" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedavg.py --gpu "5" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedprox_ft.py --gpu "4" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedprox.py --gpu "4" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python cfl.py --gpu "0" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python perfedavg.py --gpu "4" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python pfedme.py --gpu "5" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedamp.py --gpu "3" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python ditto.py --gpu "0" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedrep.py --gpu "3" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_rep_iterations $iternum --beta $beta
python pfedhn.py --gpu "5" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python fedrod.py --gpu "5" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
python knn_per.py --gpu "4" --dataset $dataset --datadir $datadir --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta

# Ours
python pfedgraph_cosine.py --gpu "1" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "all"
python pfedgraph_approx.py --gpu "1" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "all"
python pfedgraph_cosine.py --gpu "2" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "fc"
python pfedgraph_approx.py --gpu "2" --dataset $dataset --skew_class $skew_class --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta --alpha $alpha --difference_measure "fc"