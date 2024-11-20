import torch
import numpy as np
import torch.nn.functional as F
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mixup_data(self, x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        batch_size = x.size()[0]
        mix_index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[mix_index, :]  #
        mixed_target = lam * y + (1 - lam) * y[mix_index, :]

        return mixed_x, mixed_target, lam, mix_index
    else:
        lam = 1
        return x, y, lam, ...

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        if (1-lam) > lam:
            lam = 1-lam
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 定义是否使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mix_index = torch.randperm(batch_size).to(device)
    else:
        mix_index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[mix_index, :]
    y_a, y_b = y, y[mix_index]
    return mixed_x, y_a, y_b, lam, mix_index


def mixup_data_(x, y, alpha=1.0):
    # 随机生成一个 beta 分布的参数 lam，用于生成随机的线性组合，以实现 mixup 数据扩充。
    lam = np.random.beta(alpha, alpha)
    # 生成一个随机的序列，用于将输入数据进行 shuffle。
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    # 得到混合后的新图片
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # 得到混图对应的两类标签
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def sigmoid_rampup(current, rampdown_length, tat):
    """ Exponential rampdown"""
    if rampdown_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampdown_length)
        phase = 1.0 - current / rampdown_length
        return np.exp(- tat* phase * phase).astype(np.float)

def elg_turn( lamta_elg, loss_sum, loss_o, turn_loss):
    """ Exponential rampdown"""

    if loss_sum < turn_loss:
        lamta_elg1 = (turn_loss - loss_sum) / loss_o + lamta_elg
    else:
        lamta_elg1 = lamta_elg
    return lamta_elg1


def convert_label(label, num_class, e_num):
    one_hot = F.one_hot(label, num_classes=num_class)
    print(one_hot)
    one_hot = torch.where(one_hot == 1, torch.tensor(e_num).to(device), torch.tensor((1.0-e_num) / (num_class-1)).to(device))

    return one_hot

def threshold_vector(vector):
    threshold = 1.0
    replacement_small = 0.1
    replacement_large = 1.0

    # 使用PyTorch的张量操作和函数
    vector = torch.where(vector < threshold, replacement_small, replacement_large)

    return vector
