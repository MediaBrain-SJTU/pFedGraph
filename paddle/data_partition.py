import numpy as np
import random

def partition_data(partition, n_train, n_parties, train_label, beta = 0.5, skew_class = 2):
    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        
    elif partition == 'noniid':
        min_size = 0
        min_require_size = 10
        K = int(train_label.max() + 1)

        N = n_train
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(train_label == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))   # class k在所有clients上的一个分布向量
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            # print(len(net_dataidx_map[j]))
        class_dis = np.zeros((n_parties, K))

        for j in range(n_parties):
            for m in range(K):
                class_dis[j,m] = int((np.array(train_label[idx_batch[j]])==m).sum())
        # print(class_dis.astype(int))
        
    elif partition == 'noniid-skew':
        num_classes = int(train_label.max() + 1)
        num_cluster = num_classes / skew_class
        client_num_per_cluster = int(n_parties / num_cluster)
        assert num_classes % skew_class == 0, 'num_classes must be an integer multiple of skew_class'
        assert n_parties % num_cluster == 0, 'n_parties must be an integer multiple of num_cluster'
        net_dataidx_map = {i: list() for i in range(n_parties)}

        label_idx = []
        for i in range(num_classes):
            label_idx.append(list())
        for i in range(n_train):
            label_idx[int(train_label[i])].append(i)
        
        for i in range(n_parties):
            client_cluster_id = int(i // num_cluster)
            for j in range(skew_class):
                label = int(skew_class * (i % num_cluster) + j)
                sample_num_per_client = int(len(label_idx[label]) // client_num_per_cluster)
                net_dataidx_map[i] += label_idx[label][sample_num_per_client * client_cluster_id : sample_num_per_client * (1 + client_cluster_id)]
            random.shuffle(net_dataidx_map[i])
    elif partition == 'noniid-skew2':
        net_dataidx_map = {i: list() for i in range(n_parties)}
        label_idx = []
        for i in range(10):
            label_idx.append(list())
        for i in range(n_train):
            label_idx[int(train_label[i])].append(i)
        for i in range(10):
            net_dataidx_map[i] += label_idx[(i * 2) % 10][1250 * (i // 5) : 1250 * (i // 5 + 1)]
            net_dataidx_map[i] += label_idx[(i * 2) % 10 + 1][1250 * (i // 5) : 1250 * (i // 5 + 1)]
            random.shuffle(net_dataidx_map[i])
        for i in range(3):
            for j in range(5):
                net_dataidx_map[10 + i * 2] += label_idx[j][2500 + 500 * i : 2500 + 500 * (i + 1)]
                net_dataidx_map[10 + i * 2 + 1] += label_idx[j + 5][2500 + 500 * i : 2500 + 500 * (i + 1)]
        for i in range(4):
            for j in range(10):
                net_dataidx_map[16 + i] += label_idx[j][4000 + 250 * i : 4000 + 250 * (i + 1)]
        for i in range(20):
            random.shuffle(net_dataidx_map[i])
            
    traindata_cls_counts = record_net_data_stats(train_label, net_dataidx_map)
    data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:,np.newaxis]
    return net_dataidx_map, traindata_cls_counts, data_distributions
            
def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    num_classes = int(y_train.max()) + 1

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_dict[net_i] = tmp
        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate(
                        (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1,num_classes))


    data_list=[]
    for net_id, data in net_cls_counts_dict.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts_dict))

    print(net_cls_counts_npy.astype(int))
    return net_cls_counts_npy