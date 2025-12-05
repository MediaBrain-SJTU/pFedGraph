import os
import pickle
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as c_transforms
from data_partition import partition_data


def _load_cifar10_from_binary(base_path, train=True):
    """
    从本地 CIFAR-10 二进制文件读取数据，返回 numpy
    参考 paddle/tensorflow 版本的实现。
    """
    # 兼容：既支持 base_path 下直接是 cifar-10-batches-py，也支持已经进入该目录
    if os.path.exists(os.path.join(base_path, "cifar-10-batches-py")):
        cifar_root = os.path.join(base_path, "cifar-10-batches-py")
    else:
        cifar_root = base_path

    def _load_batch(file_name):
        file_path = os.path.join(cifar_root, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find CIFAR-10 batch file: {file_path}")
        with open(file_path, "rb") as f:
            dic = pickle.load(f, encoding="bytes")
        data = dic[b"data"]          # (N, 3072)
        labels = dic[b"labels"]      # list
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW -> NHWC
        data = data.astype(np.uint8)
        labels = np.array(labels, dtype=np.int64)
        return data, labels

    datas, targets = [], []
    if train:
        for i in range(1, 6):
            d, t = _load_batch(f"data_batch_{i}")
            datas.append(d)
            targets.append(t)
    else:
        d, t = _load_batch("test_batch")
        datas.append(d)
        targets.append(t)

    data = np.concatenate(datas, axis=0)
    labels = np.concatenate(targets, axis=0)
    return data, labels


def _create_mindspore_dataset(images, labels, batch_size, train, dataset_name):
    """
    使用 numpy 数据创建 mindspore.dataset，并应用与原来 torch 版本等价的增强与归一化。
    images: (N, 32, 32, 3), uint8
    labels: (N,), int64
    """
    # MindSpore 的 GeneratorDataset 需要一个可迭代对象或生成器
    def generator():
        for img, label in zip(images, labels):
            yield (img, label)

    dataset = ds.GeneratorDataset(source=generator, column_names=["image", "label"])

    # 归一化参数（按 [0, 1] 归一化）
    if dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:  # cifar100
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    trans = []
    if train:
        trans.extend([
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip()
        ])
    # 将像素缩放到 [0, 1]
    trans.append(vision.Rescale(1.0 / 255.0, 0.0))
    # 在 HWC 格式上做归一化，然后再转成 CHW
    trans.append(vision.Normalize(mean=mean, std=std))
    trans.append(vision.HWC2CHW())

    type_cast_label = c_transforms.TypeCast(np.int32)

    dataset = dataset.map(operations=trans, input_columns="image")
    dataset = dataset.map(operations=type_cast_label, input_columns="label")
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset


def cifar_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    """
    使用本地 CIFAR 数据 + numpy + mindspore.dataset 来构建各个 client 的 DataLoader（Dataset）列表。
    返回：
        train_datasets, val_datasets, test_dataset, net_dataidx_map, traindata_cls_counts, data_distributions
    其中 train/val/test 都是 mindspore.dataset 对象，可直接在训练/测试中迭代 (image, label)。
    """
    if dataset == "cifar10":
        # 先读取全部训练/测试标签，用于划分
        train_images, train_labels = _load_cifar10_from_binary(base_path, train=True)
        test_images, test_labels = _load_cifar10_from_binary(base_path, train=False)
    else:
        raise NotImplementedError("当前 mindspore 版本仅实现 CIFAR-10，如需 CIFAR-100 可按 paddle/tensorflow 版本类似扩展。")

    n_train = train_labels.shape[0]
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(
        partition, n_train, n_parties, train_labels, beta, skew_class
    )
    
    train_datasets = []
    val_datasets = []

    for i in range(n_parties):
        idxs = net_dataidx_map[i]
        np.random.shuffle(idxs)
        split = int(0.8 * len(idxs))
        train_idxs = idxs[:split]
        val_idxs = idxs[split:]

        train_ds = _create_mindspore_dataset(
            images=train_images[train_idxs],
            labels=train_labels[train_idxs],
            batch_size=batch_size,
            train=True,
            dataset_name=dataset,
        )
        val_ds = _create_mindspore_dataset(
            images=train_images[val_idxs],
            labels=train_labels[val_idxs],
            batch_size=batch_size,
            train=False,
            dataset_name=dataset,
        )

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    
    test_dataset = _create_mindspore_dataset(
        images=test_images,
        labels=test_labels,
        batch_size=batch_size,
        train=False,
        dataset_name=dataset,
    )

    return train_datasets, val_datasets, test_dataset, net_dataidx_map, traindata_cls_counts, data_distributions
    