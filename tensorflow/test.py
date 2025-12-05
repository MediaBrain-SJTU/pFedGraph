import numpy as np
import tensorflow as tf

def compute_local_test_accuracy(model, dataloader, data_distribution):
    # Keras 模型默认处于 inference 模式，除非调用 fit/train_on_batch
    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    
    generalized_total, generalized_correct = 0, 0
    
    for x, target in dataloader:
        out = model(x, training=False)
        pred_label = tf.argmax(out, axis=1)
        target = tf.cast(target, tf.int64)
        
        correct_filter = (pred_label == target)
        generalized_total += x.shape[0]
        generalized_correct += np.sum(correct_filter.numpy())
        
        target_np = target.numpy()
        correct_np = correct_filter.numpy()
        
        for i, true_label in enumerate(target_np):
            total_label_num[true_label] += 1
            if correct_np[i]:
                correct_label_num[true_label] += 1
                
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()
    
    # 防止除零错误
    if personalized_total == 0: personalized_total = 1e-10
    if generalized_total == 0: generalized_total = 1e-10
    
    return personalized_correct / personalized_total, generalized_correct / generalized_total

def compute_acc(net, test_data_loader):
    # net is Keras model
    correct, total = 0, 0
    for x, target in test_data_loader:
        out = net(x, training=False)
        pred_label = tf.argmax(out, axis=1)
        total += x.shape[0]
        correct += np.sum((pred_label.numpy() == target.numpy()))
    return correct / float(total)

def evaluate_global_model(args, nets_this_round, global_model, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    # 将 benign_client_list 转换为 list 确保兼容性
    benign_list = list(benign_client_list)
    
    for net_id, _ in nets_this_round.items():
        if net_id in benign_list:
            val_local_dl = val_local_dls[net_id]
            data_distribution = data_distributions[net_id]

            val_acc = compute_acc(global_model, val_local_dl)
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(global_model, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
    
    return np.array(best_test_acc_list)[np.array(benign_list)].mean()