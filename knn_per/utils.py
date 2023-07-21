import torch
import numpy as np


def obtain_feature_label_pair(model, train_dataloader, num_extract=5000):
    model.eval()
    model.cuda()
    features, labels = [], []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            rep = model.base(x)
            features.append(rep)
            labels.append(target)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    min_num = num_extract if num_extract < len(labels) else len(labels)
    random_seq = torch.randperm(len(labels))[:min_num]
    return features[random_seq], labels[random_seq]

def return_knn_pred(l2_distance, template_labels, n_classes, knn_k):
    pred = torch.zeros((l2_distance.shape[0], n_classes)).cuda()
    values, indices = torch.topk(l2_distance, k=knn_k, dim=1, largest=True, sorted=True)
    for sample_id in range(l2_distance.shape[0]):
        topk_labels = template_labels[indices[sample_id]]
        for k_id in range(knn_k):
            pred[sample_id, topk_labels[k_id]] += torch.exp(values[sample_id, k_id])
    return torch.softmax(pred, dim=1)

def compute_knn_acc_test(model, dataloader, data_distribution, feature_label_pair, n_classes, knn_k, interpolation):

    model.eval()
    toatl_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    template_features, template_labels = feature_label_pair
    num_features = template_features.shape[0]
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            model_pred = torch.softmax(out, dim=1)

            feature = model.base(x)
            if len(feature.size())<2:
                feature = feature.unsqueeze(0)
            feature = feature.unsqueeze(1).repeat(1,num_features,1)
            template = template_features.unsqueeze(0).repeat(feature.shape[0],1,1)
            l2_distance = -torch.pow((feature-template), 2)
            l2_distance = torch.sum(l2_distance, dim=-1)
            knn_pred = return_knn_pred(l2_distance, template_labels, n_classes, knn_k)

            final_pred = interpolation * knn_pred + (1.0-interpolation) * model_pred
            _, pred_label = torch.max(final_pred, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution).sum()
    
    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total

def compute_knn_acc_val(net, test_data_loader, feature_label_pair, n_classes, knn_k, interpolation):
    net.eval()
    correct, total = 0, 0
    template_features, template_labels = feature_label_pair
    num_features = template_features.shape[0]
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)        # model output
            model_pred = torch.softmax(out, dim=1)

            feature = net.base(x)
            if len(feature.size())<2:
                feature = feature.unsqueeze(0)
            feature = feature.unsqueeze(1).repeat(1,num_features,1)
            template = template_features.unsqueeze(0).repeat(feature.shape[0],1,1)
            l2_distance = -torch.pow((feature-template), 2)
            l2_distance = torch.sum(l2_distance, dim=-1)
            knn_pred = return_knn_pred(l2_distance, template_labels, n_classes, knn_k)

            final_pred = interpolation * knn_pred + (1.0-interpolation) * model_pred
            _, pred_label = torch.max(final_pred, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    return correct / float(total)

def local_evaluation(args, nets_this_round, global_model, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, n_classes, benign_client_list):

    for net_id, _ in nets_this_round.items():
        if net_id in benign_client_list:
            feature_label_pair = obtain_feature_label_pair(global_model, train_local_dls[net_id])
            val_acc = compute_knn_acc_val(global_model, val_local_dls[net_id], feature_label_pair, n_classes, args.knn_k, args.interpolation)
            personalized_test_acc, generalized_test_acc = compute_knn_acc_test(global_model, test_dl, data_distributions[net_id], feature_label_pair, n_classes, args.knn_k, args.interpolation)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        global_model.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()