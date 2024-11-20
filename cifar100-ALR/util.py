import torch
import numpy as np
import torch.nn as nn
import os



def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mix_index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[mix_index]
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

def pearson(x,y):
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)

    diff_x = x - mean_x
    diff_y = y - mean_y

    std_x = torch.std(x, dim=1, keepdim=True)
    std_y = torch.std(x, dim=1, keepdim=True)

    correlation = torch.sum(diff_x * diff_y, dim=1) #/ (std_x*std_y)
    return correlation


def threshold_vector(vector):
    threshold = 0.1
    replacement_small = 0.01
    replacement_large = 0.1

    # 使用PyTorch的张量操作和函数
    vector = torch.where(vector < threshold, replacement_small, replacement_large)

    return vector

# model-related
def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias.data, val=0)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, val=0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)