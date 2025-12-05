import paddle
from PIL import Image
import paddle.io as data
from paddle.io import DataLoader, Dataset
import paddle.vision.transforms as transforms
from paddle.vision.datasets import Cifar100
from PIL import Image
import numpy as np
import pickle
import os
import os.path
import logging
from data_partition import partition_data

# 设置日志
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# === 1. 辅助函数：直接读取二进制文件 ===
class ToPILImage(object):
    """
    自定义的 ToPILImage 变换。
    将 Numpy 数组 (H, W, C) 转换为 PIL Image。
    """
    def __call__(self, pic):
        # 确保输入是 numpy 数组或 Tensor，转换为 PIL
        if isinstance(pic, np.ndarray):
            return Image.fromarray(pic)
        return pic

def load_cifar10_batch(file):
    """读取 CIFAR-10 单个 batch（二进制）"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')

    # 数据 shape: (10000, 3072)
    data = dict['data']
    labels = dict['labels']

    # reshape -> (N, 3, 32, 32) -> 再转 (N, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype(np.uint8)

    return data, np.array(labels, dtype=np.int64)

def load_cifar_from_binary(root, train=True):
    """用于在 partition 之前快速获取所有数据和标签"""
    # 兼容性处理：检查 root 是否包含 cifar-10-batches-py，或者 root 本身就是
    if os.path.exists(os.path.join(root, 'cifar-10-batches-py')):
        base = os.path.join(root, 'cifar-10-batches-py')
    else:
        base = root

    def load_file(f):
        fpath = os.path.join(base, f)
        if not os.path.exists(fpath):
             raise FileNotFoundError(f"Binary file not found: {fpath}")
        with open(fpath, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic[b'data'], dic[b'labels']

    data_list, label_list = [], []

    if train:
        for i in range(1, 6):
            data, labels = load_file(f"data_batch_{i}")
            data_list.append(data)
            label_list.extend(labels)
    else:
        data, labels = load_file("test_batch")
        data_list.append(data)
        label_list.extend(labels)

    data = np.vstack(data_list).reshape(-1, 3, 32, 32)
    data = np.transpose(data, (0, 2, 3, 1))  # CHW → HWC
    data = data.astype(np.uint8)

    target = np.array(label_list, dtype=np.int64)

    return data, target

# === 2. Dataset 类定义 ===

class Cifar10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        # 自动处理路径：如果 root 下有 cifar-10-batches-py 则进入，否则假设 root 就是数据目录
        if os.path.exists(os.path.join(root, "cifar-10-batches-py")):
            self.base_folder = os.path.join(root, "cifar-10-batches-py")
        else:
            self.base_folder = root
            
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        """直接读取 CIFAR10 二进制文件"""
        if self.train:
            batch_list = [
                "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"
            ]
        else:
            batch_list = ["test_batch"]

        # ------------ 加载全部 batch -------------
        datas = []
        labels = []

        for batch_name in batch_list:
            file_path = os.path.join(self.base_folder, batch_name)
            if not os.path.exists(file_path):
                 # 如果找不到，尝试不带 base_folder (兼容性)
                 if os.path.exists(batch_name):
                     file_path = batch_name
                 else:
                     raise FileNotFoundError(f"Cannot find data batch: {file_path}")
                     
            data, label = load_cifar10_batch(file_path)
            datas.append(data)
            labels.append(label)

        data = np.concatenate(datas, axis=0)
        target = np.concatenate(labels, axis=0)

        # ------------ 过滤 dataidxs（如果有） -------------
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        """将 index 指定样本的 G/B 通道置 0"""
        self.data[index, :, :, 1] = 0
        self.data[index, :, :, 2] = 0

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        # Paddle transform (如 ToTensor) 通常接受 PIL Image 或 numpy (H, W, C)
        # 你的 transform 中包含 ToPILImage 吗？如果不包含，且 img 是 numpy uint8，
        # 最好先转 PIL 或者确保 transform 第一步接受 numpy。
        # 这里为了稳妥，先转 PIL，这与标准 PyTorch/Paddle 行为一致
        # 如果你的 transform 列表里没有 ToPILImage，可以直接传 numpy，但通常 ToTensor 需要输入是 [0,255]
        
        # 保持你原始代码的逻辑，直接传 numpy 给 transform
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class Cifar100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # 使用 Paddle 官方 API 获取基础数据
        # mode='train' or 'test'
        mode = 'train' if self.train else 'test'
        Cifar_dataobj = Cifar100(mode=mode, backend='cv2') # backend='cv2' 也就是不转 PIL，保留 numpy

        # 获取数据和标签
        data = Cifar_dataobj.data
        target = np.array(Cifar_dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        
        # Cifar100_truncated 这里你保留了转 PIL 的逻辑
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.data)


# === 3. 主加载函数 ===

def cifar_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    # 1. 预先加载所有训练数据以计算分区 (Partition)
    #    我们需要拿到完整的 y_train 来进行划分
    if dataset == "cifar10":
        # 使用快速二进制加载获取完整数据和标签
        print(f"Loading raw CIFAR-10 data from {base_path} for partitioning...")
        _, y_train = load_cifar_from_binary(base_path, train=True)
        
        # 定义 Transform
        transform_train = transforms.Compose([
            ToPILImage(), # 因为 Cifar10_truncated 返回 numpy，这里加一步转 PIL 以配合后续 transform
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([
            ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            
    elif dataset == "cifar100":
        print(f"Loading raw CIFAR-100 data from {base_path} for partitioning...")
        # 临时加载一次官方数据集获取标签
        temp_train = Cifar100(mode='train')
        y_train = np.array(temp_train.labels)
        
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            # Cifar100_truncated 中已经做了 Image.fromarray，所以这里不需要 ToPILImage
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    
    n_train = y_train.shape[0]
    
    # 2. 计算数据划分索引
    print(f"Partitioning data (n={n_train}) into {n_parties} parties...")
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, y_train, beta, skew_class)
    
    train_dataloaders = []
    val_dataloaders = []
    
    # 3. 构建各个 Client 的 DataLoader
    print("Building DataLoaders...")
    for i in range(n_parties):
        idxs = net_dataidx_map[i]
        np.random.shuffle(idxs) # 打乱索引
        
        # 划分本地训练集和验证集 (80/20)
        split = int(0.8 * len(idxs))
        train_idxs = idxs[:split]
        val_idxs = idxs[split:]
        
        if dataset == "cifar10":
            train_ds = Cifar10_truncated(root=base_path, dataidxs=train_idxs, train=True, transform=transform_train)
            val_ds = Cifar10_truncated(root=base_path, dataidxs=val_idxs, train=True, transform=transform_test)
        elif dataset == "cifar100":
            train_ds = Cifar100_truncated(root=base_path, dataidxs=train_idxs, train=True, transform=transform_train)
            val_ds = Cifar100_truncated(root=base_path, dataidxs=val_idxs, train=True, transform=transform_test)
            
        train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
    
    # 4. 构建全局测试集
    if dataset == "cifar10":
        test_ds = Cifar10_truncated(root=base_path, train=False, transform=transform_test)
    elif dataset == "cifar100":
        test_ds = Cifar100_truncated(root=base_path, train=False, transform=transform_test)
        
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions