import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.datasets import cifar10, cifar100
from data_partition import partition_data

# === 新增：本地数据加载辅助函数 ===
def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data."""
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]
    
    # 原始数据格式是 (N, 3, 32, 32)，转换为 TF 需要的 (N, 32, 32, 3)
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = data.transpose(0, 2, 3, 1) # NCHW -> NHWC
    
    return data, labels

def load_cifar10_local(path):
    """Load CIFAR10 from local path."""
    # 也可以直接是 base_path 下，这里做个兼容检查
    if os.path.exists(os.path.join(path, 'cifar-10-batches-py')):
        path = os.path.join(path, 'cifar-10-batches-py')
        
    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 32, 32, 3), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Could not find {fpath}. Please check your base_path.")
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath, label_key='labels')

    fpath = os.path.join(path, 'test_batch')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Could not find {fpath}.")
    x_test, y_test = load_batch(fpath, label_key='labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (x_train, y_train), (x_test, y_test)

def load_cifar100_local(path):
    """Load CIFAR100 from local path."""
    if os.path.exists(os.path.join(path, 'cifar-100-python')):
        path = os.path.join(path, 'cifar-100-python')
        
    fpath_train = os.path.join(path, 'train')
    fpath_test = os.path.join(path, 'test')
    
    if not os.path.exists(fpath_train) or not os.path.exists(fpath_test):
         raise FileNotFoundError(f"Could not find CIFAR100 files in {path}")

    x_train, y_train = load_batch(fpath_train, label_key='fine_labels')
    x_test, y_test = load_batch(fpath_test, label_key='fine_labels')
    
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    
    return (x_train, y_train), (x_test, y_test)
# =================================

# 数据增强函数 (保持不变)
def augment(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) 
    image = tf.image.resize_with_crop_or_pad(image, 32 + 4, 32 + 4)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image = (image - mean) / std
    return image, label

def preprocess_test(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image = (image - mean) / std
    return image, label

# CIFAR100 Normalization (保持不变)
def augment_c100(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 32 + 4, 32 + 4)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    image = (image - mean) / std
    return image, label

def preprocess_test_c100(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    image = (image - mean) / std
    return image, label

def cifar_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    # 修改逻辑：优先尝试从 base_path 加载，如果失败或未指定则回退到 keras 默认
    loaded_local = False
    if base_path and os.path.exists(base_path):
        try:
            if dataset == "cifar10":
                print(f"Loading CIFAR10 from local path: {base_path}")
                (x_train, y_train), (x_test, y_test) = load_cifar10_local(base_path)
                aug_func = augment
                test_func = preprocess_test
                loaded_local = True
            elif dataset == "cifar100":
                print(f"Loading CIFAR100 from local path: {base_path}")
                (x_train, y_train), (x_test, y_test) = load_cifar100_local(base_path)
                aug_func = augment_c100
                test_func = preprocess_test_c100
                loaded_local = True
        except Exception as e:
            print(f"Failed to load from local path {base_path}: {e}. Falling back to Keras default.")

    if not loaded_local:
        if dataset == "cifar10":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            aug_func = augment
            test_func = preprocess_test
        elif dataset == "cifar100":
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            aug_func = augment_c100
            test_func = preprocess_test_c100
        
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    n_train = y_train.shape[0]
    # data_partition
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, y_train, beta, skew_class)
    
    train_dataloaders = []
    val_dataloaders = []
    
    for i in range(n_parties):
        idxs = net_dataidx_map[i]
        # 80/20 split
        split = int(0.8 * len(idxs))
        train_idxs = idxs[:split]
        val_idxs = idxs[split:]
        
        # Train Dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train[train_idxs], y_train[train_idxs]))
        train_ds = train_ds.shuffle(10000).map(aug_func, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Val Dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_train[val_idxs], y_train[val_idxs]))
        val_ds = val_ds.map(test_func, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_dataloaders.append(train_ds)
        val_dataloaders.append(val_ds)
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(test_func, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataloaders, val_dataloaders, test_ds, net_dataidx_map, traindata_cls_counts, data_distributions