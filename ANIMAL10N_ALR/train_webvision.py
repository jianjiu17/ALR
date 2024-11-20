import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 定义是否使用GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import random
from resnet import ResNet, ResidualBlock
from classnet import classNet
from resnet import ResNet181
from resnet34 import ResNet34
from cifar10_data import CIFAR10RandomLabels
from dataloader_clothing1M import clothing_dataloader
from newSGD import newSGD
from collections import OrderedDict
import numpy
import numpy as np
import torch.nn.functional as F
from pytorch_metric_learning import losses
import math
from imageio import imwrite
import copy
from util import mixup_data, sigmoid_rampup, threshold_vector
import csv
import torchvision.models as models
from dataloader_webvision import webvision_dataloader
from InceptionResNetV2 import InceptionResNetV2

def seed_torch(seed=123):
    random.seed(seed)  # python seed
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(seed)  # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)  # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    # torch.backends.cudnn.enabled = True # pytorch使用CUDANN加速，即使用GPU加速


seed_torch(seed=345)


# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--outf', default='/home/user_1/zhangwenzhen/resnet-Webvision/model',
                    help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='/home/user_1/zhangwenzhen/resnet-Webvision/model/Resnet18.pth',
                    help="path to net (to continue training)")  # 恢复训练时的模型路径
parser.add_argument('--data_path', default='/home/user_1/zhangwenzhen/google_resized_256/', type=str, help='path to dataset')
parser.add_argument('--seed', default=346)
parser.add_argument('--num_classes', default=50, type=int)
parser.add_argument('--boost', default=50, type=int)
parser.add_argument('--early_stop', default=40, type=int, help='early stop epoch')
parser.add_argument('--dataset', default='webvision', type=str)
parser.add_argument('--model', default='InceptionResNetV2_1e-3', type=str)
parser.add_argument('--tosharp', default='yesdecline', type=str)
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--num_train', default=65944, type=int, help='train size')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--alpha', default=0.9, type=float, help='alpha in time_integrated')
parser.add_argument('--lr', default=0.01, type=float, help='lr_sgd')
parser.add_argument('--sharp_rate', default=0.2, type=float, help='sp in sharp rate')
parser.add_argument('--logs_path', default='classification_logs/',
                    help='folder to output images and model checkpoints')  # 输出结果保存
args = parser.parse_args()


# ===================================================================
def save_csv(log_dir, statistics_file_name, list_of_statistics, create=False):
    if create:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)
    else:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)


def build_experiment_folder(log_path):
    logs_filepath = "{}".format(log_path)

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)

    return logs_filepath


clas = args.num_classes
rat_avg = 0.02
# ===================================================================

mse_loss = torch.nn.MSELoss(reduction='none')


# ==================================================软标签交叉熵
def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


# loss_soft = SoftCrossEntropy()

# ====================================================================
class HardBootstrappingLoss(nn.Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """

    def __init__(self, beta=0.8, reduce=True):  # True  False
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        # beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')
        # z = argmax(p)
        z_max = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        zr, _ = F.softmax(y_pred.detach(), dim=1).max(dim=1)
        print('zzzz', z_max)
        t = y.view(-1, 1)

        # ===========================================================================
        fy = F.softmax(y_pred.detach(), dim=1).gather(1, t).view(-1)
        s3 = (fy < rat_avg).type(torch.int)
        # print('s3', s3.sum())

        s1 = s3  # s2 * s3

        # print('tr', tr)
        s = s1.detach()  # s1 选出来的噪声
        print('s', s, s.sum())
        z = s * z_max + (torch.ones_like(s) - s) * y

        # ===============生成标签   生成标签     生成标签===================
        max_10, max10_d = F.softmax(y_pred.detach(), dim=1).kthvalue(clas, dim=1)  # _d表示 第n大的值的类别
        max_9, max9_d = F.softmax(y_pred.detach(), dim=1).kthvalue(clas - 1, dim=1)
        max_8, max8_d = F.softmax(y_pred.detach(), dim=1).kthvalue(clas - 2, dim=1)

        max10_doh = F.one_hot(max10_d, num_classes=clas)  # 标签one-hot化
        max9_doh = F.one_hot(max9_d, num_classes=clas)
        max8_doh = F.one_hot(max8_d, num_classes=clas)
        # max3_num = max10_doh + max9_doh + max8_doh
        max_sum = max_10 + max_9 + max_8
        max10_a = (max_10 / max_sum).unsqueeze(1).repeat(1, clas)
        max9_a = (max_9 / max_sum).unsqueeze(1).repeat(1, clas)
        max8_a = (max_8 / max_sum).unsqueeze(1).repeat(1, clas)
        max3_num = max10_doh * max10_a + max9_doh * max9_a + max8_doh * max8_a
        # fy = F.softmax(y_pred.detach(), dim=1).gather(1, t).view(-1)
        # s3 = (fy < 0.1).type(torch.int)
        ssoft = torch.unsqueeze(s1, 1)
        ssoft = ssoft.repeat(1, clas)
        max_soft = max3_num * ssoft  # 前三个标签作为噪声的软标签
        t_hard = ((torch.ones_like(s1) - s1).unsqueeze(1).repeat(1, clas)) * F.one_hot(y, num_classes=clas)

        t_all = max_soft.detach() + t_hard.detach()
        # print('t_all::', t_all)
        # ==================生成标签   生成标签    生成标签========================

        # bootstrap = (1.0 - self.beta)*F.cross_entropy(y_pred, z, reduction='none')
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')
        # bootstrap = F.cross_entropy(y_pred, z, reduction='none')
        # print('xinxishang::', torch.mean(- F.softmax(y_pred.detach(), dim=1)*torch.log2(F.softmax(y_pred.detach(), dim=1))))
        # second term = (1 - beta) * z * log(p)

        bootstrap = SoftCrossEntropy(y_pred, t_all)
        if self.reduce:
            # return torch.mean(bootstrap), s1
            return bootstrap, s1, s * z_max + (torch.ones_like(s) - s) * (-1), s * max9_d + (torch.ones_like(s) - s) * (
                -1), s * max8_d + (torch.ones_like(s) - s) * (-1), (
                           torch.ones_like(s) - s).sum(), s * max_9  # bootstrap

            # return bootstrap.sum()/(torch.ones_like(s) - s).sum() , s1    #s1 选出来的噪声
        return beta_xentropy + bootstrap


# ======================================================================
class HardBootstrappingLoss1(nn.Module):
    def __init__(self, beta=0.8, reduce=True):  # True  False
        super(HardBootstrappingLoss1, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y, y_cs, epoch_num=0):
        zmax = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        t = y.view(-1, 1)
        t1 = y_cs.view(-1, 1)
        max_1, max1_mun = F.softmax(y_pred.detach(), dim=1).kthvalue(clas, dim=1)
        max_2, max2_mun = F.softmax(y_pred.detach(), dim=1).kthvalue(clas - 1, dim=1)
        max_3, max3_mun = F.softmax(y_pred.detach(), dim=1).kthvalue(clas - 2, dim=1)
        max_4, max4_mun = F.softmax(y_pred.detach(), dim=1).kthvalue(clas - 3, dim=1)
        max_5, max5_mun = F.softmax(y_pred.detach(), dim=1).kthvalue(clas - 4, dim=1)
        max_7, max7_mun = F.softmax(y_pred.detach(), dim=1).kthvalue(clas - 6, dim=1)
        max_10, max10_mun = F.softmax(y_pred.detach(), dim=1).kthvalue(1, dim=1)
        print('max_10', max10_mun)
        fy = F.softmax(y_pred.detach(), dim=1).gather(1, t).view(-1)
        fy_cs = F.softmax(y_pred.detach(), dim=1).gather(1, t1).view(-1)
        s1 = (fy < rat_avg).type(torch.int)



        # s = s1
        y_soft = F.softmax(y_pred.detach(), dim=1)
        y_soft_max, indexd = y_soft.max(dim=1)
        # ====================================================

        fy_cs_num = (fy_cs <= max_7).sum()

        z_at = np.array(y.detach().data.cpu())
        z1_s, _ = noisify_multiclass_symmetric(z_at, noise=0.5)  # 随机从其余类别选择一个作为标签
        z1_s = torch.from_numpy(z1_s).to(device)

        # ==================================
        '''max1_mun_np = np.array(max1_mun.data.cpu())
        max1_mun1, _ = noisify_multiclass_symmetric(max1_mun_np, noise=0.0)  # 随机从其余类别选择一个作为标签
        max1_mun1 = torch.from_numpy(max1_mun1).to(device)'''
        if epoch_num > 2000:  # 20

            s = s1  # * s_val #* spz_mask#* (1 - (fy >= max_2).type(torch.int))# 0.15<max_2,  max_1< 0.5
            print('ssssssssssssssssssssssssssssssssss', s.sum())
            # z1 = s * max1_mun + (s1-s) * z1_s + (torch.ones_like(s1) - s1) * y
            z1 = s * max1_mun + (torch.ones_like(s) - s) * y
            tk = (s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) * (max1_mun == y_cs) + (
                        torch.ones_like(s) - s) * (max1_mun == y) * (
                              max1_mun == y_cs)).sum()  # ((max1_mun == y_cs) * (max_1 / max_2 > max_2 / max_3) * (max_1 / max_2 > max_2 / max_10)).sum()
            tk2 = (s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) + (torch.ones_like(s) - s) * (
                        max1_mun == y)).sum()  # ((max_1 / max_2 > max_2 / max_3) * (max_1 / max_2 > max_2 / max_10)).sum()
            print('fy===last::', ((fy < max_7) * s * (max1_mun == y_cs)).sum())  # 认定的噪声中预测是对的的样本的噪声标签小于max_7  ,15%
            print('fy===last3::', ((fy > max_4) * s * (max1_mun == y_cs)).sum())  # 认定的噪声中预测是对的的样本的噪声标签小于max_7  ,15%
            print('fy===last4-7::',
                  ((fy >= max_7) * (fy <= max_4) * s * (max1_mun == y_cs)).sum())  # 认定的噪声中预测是对的的样本的噪声标签小于max_7  ,15%

            bootstrap = F.cross_entropy(y_pred, z1,
                                        reduction='none') * max_1.detach()  # * torch.from_numpy(ycc_bh*2.0 + (1.0-ycc_bh)*0.5).to(device) #(torch.ones_like(s)-(s*(max_1 <= k_7)))#* max_1.detach()   #* (s*0.4 + (torch.ones_like(s) - s) *fy.detach())

        else:
            s = s1  # * s_val #* spz_mask#* (1 - (fy > max_2).type(torch.int))
            print('ssssssssssssssssssssssssssssssssss', s.sum())
            y_p_s = y_soft_max < 1.0  # 0.9*(epoch_num/10)
            y_local = torch.ones_like(s) - s * y_p_s

            if epoch_num > 50:  # 50
                y_p_s = y_soft_max < 0.95
                y_local = torch.ones_like(s) - s * y_p_s - (indexd != y) * (torch.ones_like(s) - s)

            z3 = s * max3_mun + (torch.ones_like(s) - s) * y
            z2 = s * max2_mun + (torch.ones_like(s) - s) * y

            # z1 = s * max1_mun + (s1-s) * z1_s + (torch.ones_like(s1) - s1) * y    #max1_mun1

            z1 = s * max1_mun + (torch.ones_like(s) - s) * y
            # tk = (s * (max1_mun == y_cs) * (max_2 < 0.1)).sum() + ((torch.ones_like(s) - s) * (max1_mun == y) * (max1_mun == y_cs)).sum()
            tk = ((max1_mun == y_cs) * (
                        s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) + (torch.ones_like(s) - s) * (
                            max1_mun == y))).sum()
            tk2 = (s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) + (torch.ones_like(s) - s) * (
                        max1_mun == y)).sum()  # ((max_1 / max_2 > max_2 / max_3) * (max_1 / max_2 > max_2 / max_10)).sum()
            print('fy===last::', ((fy < max_7) * s * (max1_mun == y_cs)).sum())
            # bootstrap = F.cross_entropy(y_pred, z1, reduction='none') * fy.detach()**4.5# * torch.from_numpy(ycc_bh * 1.0 + (1.0 - ycc_bh) * 0.5).to(device)
            # z2 = F.one_hot(z1, num_classes=10).type(torch.float)
            # msel = mse_loss(y_pred, z2).mean(dim=1)
            # mse_boot = msel * s + (torch.ones_like(s) - s) * bootstrap

            bootstrap = F.cross_entropy(y_pred, z1,
                                        reduction='none') * y_local  # fy.detach()   #* (torch.ones_like(s)-(s*(max_1 <= k_7)))             #* torch.pow(fy.detach(), 1)
            print('s*y_p_s::s*y_p_s::s*y_p_s::', y_local.sum(), (torch.ones_like(s) - s * y_p_s).sum(),
                  (s * y_p_s).sum())
        # bootstrap2 = F.cross_entropy(y_pred, z2, reduction='none')
        # bootstrap3 = F.cross_entropy(y_pred, z3, reduction='none')

        '''elif max_1 < 0.6:
            if max_2 > (max_1 / 2):
                lt = max_1
            lz = '''
        '''y_sym = s * (max_2 < 0.1) * (3 * max_2 < (1 - max_1)) * max1_mun + (torch.ones_like(s) - s) * (
                    max1_mun == y) * (3 * max_2 < (1 - max_1)) * max1_mun + (torch.ones_like(s) - (s * (max_2 < 0.1) + (torch.ones_like(s) - s) * (max1_mun == y)) * (3 * max_2 < ( 1 - max_1))) * max10_mun
        print('y_sym::', y_sym)'''
        if self.reduce:
            fan = ~((max_1 / max_2 > max_2 / max_3) * (max_1 / max_2 > max_2 / max_10))
            print('zsmall',
                  (s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) + (torch.ones_like(s) - s) * (max_1 == y)).sum())
            # zsmall = z1 * (max_1/max_2 > max_2/max_3) *(max_1/max_2 > max_2/max_10) + torch.from_numpy(z1_s).to(device)*fan

            zsmall = s * max1_mun * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) + (
                        (torch.ones_like(s) - s) * (max_1 == y)) * y + (torch.ones_like(s) - (
                        s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) + (torch.ones_like(s) - s) * (
                            max_1 == y))) * z1_s  # 三部分，1是选择的噪声中的可靠的，2是正确中的可靠的，3是其余的做均匀噪声
            zao1 = (torch.where(s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) == 1, max1_mun.to(device),
                                torch.ones_like(s).long() * (-1)) == y_cs).sum()
            clean1 = (torch.where(((torch.ones_like(s) - s) * (max_1 == y)) == 1, y,
                                  torch.ones_like(s).long() * (-1)) == y_cs).sum()
            last1 = (torch.where((torch.ones_like(s) - (
                        s * (max_1 / max_2 > max_2 / max_10) * (fy < max_3) + (torch.ones_like(s) - s) * (
                            max_1 == y))) == 1, z1_s, torch.ones_like(s).long() * (-1)) == y_cs).sum()
            print('z1  clean raze', zao1, clean1, last1, (zsmall == y_cs).sum(), (z1 == y_cs).sum())
            print('z1', z1)

            return torch.mean(
                bootstrap), fy_cs_num, max_1, max_2, max_3, max_4, max_5, max1_mun, tk, tk2, z1, z1, s  # zsmall, z1       bootstrap
            # return torch.sum(bootstrap)/y_local.sum(), fy_cs_num, max_1, max_2, max_3, max_4, max_5, max1_mun, tk, tk2, z1, z1, s  # zsmall, z1       bootstrap

        return bootstrap


# ======================================================================
class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=1.5):
        super(FocalLoss, self).__init__()

        self.gamma = gamma

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = nn.functional.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        return batch_loss


# ====================================================================================
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    m = y.shape[0]  # m = 128
    new_y = y.copy()  # 复制一个y
    flipper = np.random.RandomState(random_state)  # 做一个随机数种子

    for idx in np.arange(m):
        # i = y[idx]
        i = int(y[
                    idx])  # there is problem of decimals in imported yy,which is also the modified label, int transpose is adopted here!!!
        # draw a vector with only an 1   画一个只有1的向量
        flipped = flipper.multinomial(1, P[i, :], 1)[0]  # torch.multinominal方法可以根据给定权重对数组进行
        # 多次采样，返回采样后的元素下标
        new_y[idx] = np.where(flipped == 1)[0]  # 返回非零元素的下标

    return new_y


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=clas):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))  # 1*1矩阵
    n = noise
    P = (n / (nb_classes - 1)) * P  # 噪声比/类别数-1再乘上1*1矩阵

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n  # 让0.0
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n  # 让9-9
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    return y_train, actual_noise


# ===============================================================================
class GCE_loss(nn.Module):
    def __init__(self, q=0.8, xt='gd'):
        super(GCE_loss, self).__init__()
        self.q = q
        self.xt = xt

    def forward(self, outputs, targets):
        targets = torch.zeros(targets.size(0), clas).cuda().scatter_(1, targets.view(-1, 1), 1)
        pred = F.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        pred_y = torch.clamp(pred_y, 1e-4)
        if self.xt == 'mean':
            final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        else:
            final_loss = (1.0 - pred_y ** self.q) / self.q
        return final_loss


# ==================write csv=================================================
# def data_write_csv(file_name, datas):
#    file_csv = codecs.open(file_name, 'w+', 'utf-8')
#    witer =
# =========================================================================================
def adjust_learning_rate(optimizer, epoch):  # 带有warm-up的余弦学习率衰减
    """Decay the learning rate based on schedule"""
    lr = 0.02
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / 300))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# tau=0.97
"""def get_scheduler(optimizer, args):
    if args.lr_schedule == 'step':
        print("Using `step` schedule")
        return lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=args.lr_gamma)

    elif args.lr_schedule == 'cosine':
        print("Using `cosine` schedule")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    else:
        raise KeyError("LR schedule `{}` is not supported.".format(args.lr_schedule))"""


# ============ =define label transformation function=========================
def random_label_transform(labels):
    random_numbers = torch.rand(labels.shape)
    masks = random_numbers < 0.5
    random_labels = torch.randint(0, clas, labels.shape)
    transformd_labels = torch.where(masks, random_labels, labels)
    return transformd_labels


# 超参数设置
EPOCH = args.num_epochs  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = args.batch_size  # 批处理尺寸(batch_size)
# 学习率
lp = 2.4
arry = numpy.zeros(clas)
ltr = 1.1
ws = 1.0
top_k = 40
epoch_len = clas



# 训练，组成batch的时候顺序打乱取

#=========webvision_data  loader===================================================================================
loader =webvision_dataloader(batch_size=args.batch_size, num_class=args.num_classes,
                                        num_workers=8, root_dir=args.data_path)

trainloader, noisy_labels = loader.run('train')

#train_eval_loader = loader.run('eval_train')
testloader = loader.run('test')
#imagenet_valloader = loader.run('imagenet')
#train_len = loader.run('ceshi')#65944
#print("++++++++++++++++++",train_len)



#================================================================================================================
#
# 模型定义-ResNet
#net = ResNet34()
# net = ResNet181()

net = InceptionResNetV2()
# net = ResNet(ResidualBlock)
# net_class = classNet()


'''net = ResNet(ResidualBlock)
net_0 = ResNet(ResidualBlock)

#net_0.load_state_dict(torch.load('/home/user_1/zhangwenzhen/ResNet-18-cifar10/net_058.pth'), strict=True) #['state_dict']   #获得之前训练得到的参数模型
net.load_state_dict(torch.load('/home/user_1/zhangwenzhen/ResNet-18-cifar10/net_119.pth'), strict=True)'''
'''net_net = net.state_dict()
#print('net_0_0: ', net_net)
net_0_0 = net_0.state_dict()
#print('net_0_0: ', net_0_0)
#net_0_dict =  {k: v for k, v in net_0_0.items() if k in ['fc.weight']}
net_0_dict =  {k: v for k, v in net_0_0.items() if k in {'fc.weight', 'fc.bias','fc1.weight', 'fc1.bias'}}  # 'layer3.0.left.5.weight'    if k not in {'fc.weight', 'fc.bias'}
net_net.update(net_0_dict)
#print('net_net_up', net_net)
net.load_state_dict(net_net)'''

net = net.to(device)
# net_class = net_class.to(device)
# 定义损失函数和优化方式
criterion_1 = nn.CrossEntropyLoss(reduction='none')  # 损失函数为交叉熵，多用于多分类问题
criterion = nn.CrossEntropyLoss()

# criterion = GCE_loss(xt ='mean')
criterion_nom = GCE_loss()
focal = FocalLoss()
loss_func = losses.CircleLoss()
loss_HB = HardBootstrappingLoss()
loss_HB1 = HardBootstrappingLoss1()

loss_MAE = torch.nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# optimizer = newSGD(net.parameters(), lr=0.1, momentum=0.9, tau=0.8, weight_decay=5e-4)

# optimizer = optim.SGD([{'params': net.parameters()}, {'params': net_class.parameters()}], lr = 0.1, momentum= 0.98, weight_decay = 5e-4)
# optimizer = newSGD([{'params': net.parameters()}, {'params': net_class.parameters()}], lr = 0.1, momentum=0.9, tau=0.8, weight_decay=1e-3)
'''for name, param in net_class.named_parameters():
    if (name.startswith('fc')):
        param.requires_grad = False
    #if name.startswith('fc1'):
    #    param.requires_grad = True

    #print('name: ', name, param.requires_grad, param)'''
# 训练
# if __name__ == "__main__":
# seed_torch(123)
epoch_save = -1
y_file = '/home/user_1/zhangwenzhen/resnet-18-cifar100-ce/newlab2/' + "y.npy"
y_c_file = '/home/user_1/zhangwenzhen/resnet-18-cifar100-ce/labcount1/' + "y_121.npy"  # 存储每轮预测为各标签的次数
# ======================================================
train_loss_n = 0.0
train_acc_n = 0.0
test_loss_n = 0.0
test_acc_n = 0.0
noise_n = 0.0
noise_pre_n = 0.0
noise_zip_n = 0.0
jiu_true_n = 0.0
pre_cs_n = 0.0
pre_lab_cs = 0.0

# ==============================================
epoch_step = [50,90]
label_red3 = torch.zeros([args.num_train, clas], dtype=torch.float).to(device)
label_red3t = torch.zeros([args.num_train, clas], dtype=torch.float).to(device)
num_examp = args.num_train

label_mix_hist = torch.zeros([num_examp, clas], dtype=torch.float).to(device)
label_rates = torch.ones([args.num_train], dtype=torch.float).to(device)
label_up = torch.ones([args.num_train], dtype=torch.float).to(device)
label_cen = torch.zeros([clas, clas], dtype=torch.float).to(device)
best_acc = 85  # 2 初始化best test accuracy
print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
# =====================================================================================================
experiment_name = "{}_batch_{}_stop_{}_sp_{}_seed_{}_alpha_{}_boost_{}_lr_{}_{}".format(args.dataset, args.batch_size, args.early_stop,
                                                           args.sharp_rate, args.seed, args.alpha, args.boost, args.lr, args.tosharp)

logs_filepath = build_experiment_folder(log_path=args.logs_path)
if epoch_save == -1:  # if this is a new experiment
    # statistics file
    save_csv(logs_filepath, experiment_name,
             ["epoch", "train_loss", "train_acc", "noise_rate", "noise_pre", "noise_zip", "jiu_10", "pre_cs",
              "pre_lab_cs", "val_loss", "val_acc", "test_loss", "test_acc"], create=True)
# ========================================================================================================

with open("acc.txt", "w") as f:
    with open("log.txt", "w") as f2:
        with open('outputs_softmax.txt', 'w') as f4:
            for epoch in range(pre_epoch, EPOCH):

                dimt_noise = 0
                noise_num = 0.0
                noise_jisuan = 0.0
                noise_zide = 0.0
                jiu_num = 0.0
                jiu_num10 = 0.0
                jiu_num9 = 0.0
                jiu_num8 = 0.0
                last4 = 0.0
                pre_max_sum = 0.0
                pre_max_max = 0.0
                pre_cs = 0.0
                pre_lab_cs = 0.0
                tk1_num = 0.0
                tk22_num = 0.0
                pred_tol_balance = 0.0
                print('epoch-one: ', epoch)
                vj = 0
                arry = numpy.zeros(clas)
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                # net_class.train()
                sum_loss = 0.0
                train_correct = 0.0
                correct_10 = 0.0
                correct_pred_label = 0.0
                total = 0.0

                epoch1 = epoch + 1

                # adjust_learning_rate(optimizer, epoch)

                if epoch1 in epoch_step:
                    lr = optimizer.param_groups[0]['lr']

                    optimizer = optim.SGD(net.parameters(), lr * 0.1, momentum=0.9, weight_decay=1e-3)  # lr=0.3
                    # optimizer = newSGD(net.parameters(), lr * 0.1, momentum=0.9, tau=0.9, weight_decay=1e-3)

                    # optimizer = optim.SGD([{'params': net.parameters()}, {'params': net_class.parameters()}], lr * 0.1, momentum = 0.98, weight_decay = 5e-4)
                    # optimizer = newSGD([{'params': net.parameters()}, {'params': net_class.parameters()}], lr * 0.1, momentum=0.982, tau=0.99, weight_decay=1e-3)
                    print('lr: ', lr)



                # for i, data in enumerate(trainloader, 0):
                print(enumerate(trainloader))
                for i, (input, label, index) in enumerate(trainloader):
                    # 准备数据
                    actual_noise = 0.0
                    print('i:', i)
                    length = len(trainloader)
                    dimt = 0
                    # inputs, labels = data
                    # inputs = input
                    # targets = target
                    # ============================================
                    label_ccc = label.data.cpu().numpy()

                    actual_noise = label.eq(label.data).cpu().sum()
                    noise_num += actual_noise

                    label_nn = np.array(label)  # self.

                    # ==============================================
                    label_t, _ = noisify_multiclass_symmetric(label_nn, noise=1.0)  # 随机从其余类别选择一个作为标签
                    print(label_t)
                    np.random.seed(1)
                    mask = np.random.rand(len(label_nn)) <= 1.0
                    rnd_label_nn = np.random.choice(clas, mask.sum())

                    label_nn[mask] = rnd_label_nn
                    label_nn = [int(x) for x in label_nn]
                    label_nn = np.array(label_nn)
                    label_nn = torch.from_numpy(label_nn)
                    # ===============================================

                    inputs, labels = input.to(device), label.to(device)  # old ---old
                    labels_soft = F.one_hot(labels, num_classes=clas)

                    labels_two = labels  # Epoch 《100，label_two = labels, 否则 label_y_sym

                    label_cs = label.to(device)

                    label_t = np.array(label_t)
                    print('label_t', label_t)
                    label_t = torch.from_numpy(label_t)
                    label_t = label_t.to(device)

                    index = index.numpy()

                    # target1 = targets.cuda(non_blocking=True)
                    # inputs_var = torch.autograd.Variable(inputs)
                    labels_var = torch.autograd.Variable(labels)
                    print('index:   ', index)

                    outputs,train_x,train_conf = net(inputs)  # old old
                    # optimizer.zero_grad()
                    #  #############    mixup    #################################################################
                    """mix_inputs, mix_lab_a, mix_lab_b, lam, mix_index = mixup_data(inputs, labels, alpha=0.5)
                    outputs = net(mix_inputs)
                    mix_pred = F.softmax(outputs, dim=1)
                    output_x_pred = F.softmax(output_x, dim=1).data.detach()
                    mix_pred = torch.clamp(mix_pred, 1e-4, 1.0 - 1e-4)
                    label_mix_hist[index] = 0.5 * label_mix_hist[index] + 0.5 * output_x_pred / (output_x_pred).sum(dim=1,
                                                                                                          keepdim=True)
                    q = lam * label_mix_hist[index] + (1 - lam) * label_mix_hist[mix_index]
                    loss_mix = lam * criterion(outputs, mix_lab_a) + (1 - lam) * criterion(outputs, mix_lab_b)
                    elg_mix = (torch.square(q.detach() - mix_pred).sum(dim=1)).mean()"""
                    ########################################################################################

                    # output = net(torch.cat((inputs,input1s), dim=0))   # last
                    # outputs, output1s = output[:int(output.shape[0] / 2)], output[int(output.shape[0] / 2):]
                    # outputs_softmax = nn.functional.softmax(outputs, 1).cpu().detach().numpy()
                    outputs_softmax, outputs_softmax_index = torch.topk(nn.functional.softmax(outputs, 1), k=10, dim=1)  # .cpu().detach().numpy()
                    np.savetxt(f4, outputs_softmax.cpu().detach().numpy(), fmt='%.4f', delimiter=",")
                    f4.write('\n %d ' % epoch1)
                    # f4.write( '%03d  %05d | %.4f' % (epoch + 1, (i + 1 + epoch * length), outputs_softmax))
                    f4.flush()

                    logsoftmax = nn.LogSoftmax(dim=1).cuda()
                    softmax = nn.Softmax(dim=1).cuda()
                    print('epoch:   ', epoch)

                    # ====================================================================
                    rel_noise = (labels_two != label_cs).type(torch.int)  # 查看真正的噪声标签是那些
                    print('rel_noise', rel_noise)
                    # =-==============================================================================

                    # ========================================
                    pred_s = F.softmax(outputs, dim=1)
                    #pred_s = torch.clamp(pred_s, 1e-4, 1.0 - 1e-4)
                    pred_detach = pred_s.data.detach()
                    index_t = index.tolist()
                    #label_red3[index_t] = F.one_hot(labels_two, num_classes=clas) * 1.000
                    if epoch1 <= args.early_stop:
                        label_red3[index_t] = F.one_hot(labels_two, num_classes=clas) * 1.000
                    else:
                        label_red3[index_t] = args.alpha * label_red3[index_t] + (1.0-args.alpha) * pred_detach #((pred_detach) / pred_detach.sum(dim=1, keepdim=True))

                    # label_red3[index_t] = 0.9* label_red3[index_t] + 0.1* ((pred_detach) / pred_detach.sum(dim=1, keepdim=True))

                    r3_max, r3_max_dim = label_red3[index_t].max(dim=1)
                    r3_max_1, r3_max_dim_1 = r3_max.max(dim=0)
                    r3_min, r3_min_dim = label_red3[index_t].min(dim=1)
                    label_red3_one = F.one_hot(r3_max_dim, num_classes=clas)

                    r3t_max, r3t_max_dim = label_red3t[index_t].max(dim=1)

                    dimt = (r3_max_dim == label.to(device)).sum()  # 预测均值等于真实标签的概率
                    r3_mix = label_red3_one * (r3_max >= r3_max.mean()).unsqueeze(1).repeat(1, clas).type(torch.float) + \
                             label_red3[index_t] * (r3_max < r3_max.mean()).unsqueeze(1).repeat(1, clas)

                    pred_lab = label_red3[index_t].detach().gather(1, labels_two.view(-1, 1)).view(-1)

                    elr_reg1 = ((1 - (label_red3[index_t] * pred_s).sum(
                        dim=1)).log()).mean()  # ((label_red3[index_t] - pred_s)**2).mean() #((1 - (label_red3[index_t] * pred_s).sum(dim=1)).log()).mean()
                    elr_reg2 = (torch.square(label_red3[index_t] - pred_s).sum(dim=1)).mean()
                    # =======================================================================================================
                    loss3, label_cs_num, pre_max, pre_max2, pre_max3, pre_max4, pre_max5, pre_max_dim, tk1, tk22, y_sym1, y_tongji, s_noise = loss_HB1(
                        outputs, labels_two, label_cs, epoch_num=epoch1)
                    # loss31, label_cs_num1, pre_max1, pre_max21, pre_max31, pre_max41, pre_max51, pre_max_dim1, tk11, tk221, y_sym11, y_tongji1, s_noise1 = loss_HB1(
                    #   outputs, label_y1, label_cs, epoch_num=epoch1)

                    r3_future = r3_max_dim.cpu().numpy()
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    pred_max, pred_max_dim = pred_detach.max(dim=1)
                    pred_max2, pred_max2_dim = pred_detach.kthvalue(clas - 1, dim=1)
                    pred_balance = (pred_max_dim != labels_two).sum()
                    pred_tol_balance += pred_balance
                    loss1 = criterion(outputs, labels_two)
                    p1 = (pred_max - pred_max2) / pred_max
                    if epoch1 <= args.boost:  # 40,0.3  ==70+
                        #loss1 = criterion(outputs, labels_two)
                        loss1 = -((label_red3[index_t] * torch.log_softmax(outputs, dim=1)).sum(dim=1)).mean()# * p1).sum() / p1.sum()#(criterion_1(outputs, labels_two)).mean()  # r3_max / r3_max.mean() *  #criterion(outputs, labels_two) + 7.0*elr_reg1#

                    else:
                        if args.tosharp == 'yesdecline':
                            epoch_rate = 1.0 - (epoch1 - args.boost) / (100-args.boost)
                        else:
                            epoch_rate = 1.0
                        """loss1 = -((label_red3[index_t] * torch.log_softmax(outputs, dim=1)).sum(dim=1)).mean() - args.sharp_rate * epoch_rate * (
                                        (F.softmax(outputs, dim=1) * torch.log_softmax(outputs, dim=1)).sum(
                                            dim=1) * p1).sum() / p1.sum()"""
                        loss1 = -((label_red3[index_t] * torch.log_softmax(outputs, dim=1)).sum(dim=1)).mean() - args.sharp_rate * epoch_rate * ((F.softmax(outputs, dim=1) * torch.log_softmax(outputs, dim=1)).sum(dim=1)).mean()  # - 3.0*((F.softmax(outputs, dim=1) * torch.log_softmax(outputs, dim=1)).sum(dim=1)).mean()
                        # loss1 =  (r3_max / r3_max.mean() * criterion_1(outputs, labels_two)).mean() + 100.0*((torch.square(label_red3[index_t]- pred_s)).sum(dim=1)).mean() - 50*((F.softmax(outputs, dim=1) * torch.log_softmax(outputs, dim=1)).sum(dim=1)).mean()#elr_reg2 # 5000 best

                    loss2, pre_noise, label_jiu, lab9, lab8, jiu_no, max911 = loss_HB(outputs,
                                                                                      labels_two) # pre_noise 是选中的噪声  labels_two = labels

                    # ==========noise_statistic==================================================================================================
                    lab_noise = F.softmax(outputs, dim=1).gather(1, labels_two.view(-1, 1)).view(-1).detach()
                    pre_noise = (lab_noise < rat_avg).type(torch.int).detach()
                    noise_zijian = pre_noise.sum()
                    noise_zide += noise_zijian  # noise_zide 是选到的噪声

                    rel_noise = (labels_two != label_cs).type(torch.int)  # 查看真正的噪声标签是那些
                    duibi_noise = (pre_noise * rel_noise).sum()  # 选中的噪声乘以真正的噪声
                    noise_jisuan += duibi_noise  # 选中的噪声乘以真正的噪声

                    max_9, max9_d = F.softmax(outputs.detach(), dim=1).kthvalue(9, dim=1)
                    max_8, max8_d = F.softmax(outputs.detach(), dim=1).kthvalue(8, dim=1)

                    label_jiu = pre_noise * F.softmax(outputs.detach(), dim=1).argmax(dim=1) + (
                            torch.ones_like(pre_noise) - pre_noise) * (-1)
                    lab9 = pre_noise * max9_d + (torch.ones_like(pre_noise) - pre_noise) * (-1)
                    lab8 = pre_noise * max8_d + (torch.ones_like(pre_noise) - pre_noise) * (-1)
                    jiu_ture = (label_jiu == label_cs).type(torch.int).sum()  # - jiu_no
                    jiu_ture9 = (lab9 == label_cs).type(torch.int).sum()  # - jiu_no
                    jiu_ture8 = (lab8 == label_cs).type(torch.int).sum()  # - jiu_no
                    jiu_3 = jiu_ture + jiu_ture9 + jiu_ture8
                    jiu_num += jiu_3
                    jiu_num10 += jiu_ture
                    jiu_num9 += jiu_ture9
                    jiu_num8 += jiu_ture8

                    # =======================================================================================

                    pre_max_sum += (
                                (pre_max2 < rat_avg) * (pre_max_dim == label_cs)).sum()  # 预测的max2小于0.1 且 预测值等于真实值的概率
                    pre_cs += (pre_max_dim == label_cs).sum()  # 预测的==真正标签的概率
                    pre_lab_cs += ((pre_max_dim == label_cs) * (labels_two == label_cs)).sum()  # 预测的==标记的==真正的标签  的概率
                    pre_max_max += ((pre_max2 < rat_avg) * (pre_max > 0.8) * (
                                pre_max_dim == label_cs)).sum()  # 预测的max2小于0.1的概率


                    loss4 = criterion((1 - outputs), label_t)
                    last4 += label_cs_num
                    tk1_num += tk1
                    tk22_num += tk22
                    '''if epoch1 / 51.0 > 1.0:
                        ep1 = 1.0
                    else:

                        ep = epoch1 / 51.0
                        ep1 = ep ** (2.0000)
                        if epoch1 <21:
                            ep2 = 0.0
                        else:
                            ep2 = 1.0


                    losses = ep1 * (ep2*loss2 + (1-ep2)*loss3)+ (1 - ep1) * loss1 + le + loss4'''

                    loss_c = outputs.sum(axis=0).pow(2).sum() / len(outputs)  # 抑制类别向一个集中
                    label_zao = labels_two.view(-1, 1)

                    k1 = F.softmax(outputs, dim=1)
                    k2 = k1.gather(1, label_zao)
                    k3 = k2.view(-1).detach()

                    # loss_mean = criterion_1(outputs, labels_two) * (k3.to(device))**2.0 * (ycc_rate**5.0)  #(F.softmax(outputs, dim=1).gather(1, label_zao).view(-1))
                    # loss_mean = criterion_1(outputs, labels_two) * (k3.to(device) ** 1.5)* (ycc_rate ** 4.5) #
                    # losses = ep1 * (loss3+loss31) * 0.5 + (1-ep1)*(loss1+loss11)*0.5 #+ ep4*le #ep1 * loss2  + (1 - ep1) * loss1 + ep4*loss3  + ep3*le
                    # losses = ep1 * loss3 + (1 - ep1) * loss1 #+ epc*loss_c

                    losses = loss1

                    # losses = loss1 #+ 7.0*elr_reg1
                    # losses = torch.mean(loss_mean) + loss1  #loss1
                    # losses = ep1*loss3 + (1-ep1)*losses0
                    optimizer.zero_grad()
                    len_data = len(inputs)  #############
                    # forward + backward
                    # outputs = net(inputs)
                    loss_cll = loss_func(outputs, labels)
                    # losses = criterion(outputs, labels_two)
                    # losses = loss_HB(outputs,labels)

                    print('epoch111: ', epoch1 / 80.0)

                    # print('loss: %.5f' % losses)

                    # loss_cll.backward()
                    losses.backward()
                    # loss_mae.backward()
                    # loss1.backward()
                    optimizer.step()

                    # for name, param in net_class.named_parameters():
                    #    print('net_class_pararm2222:  ', name, param)

                    # optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999)).step()

                    '''if epoch >= 1 and epoch < 20:
                        lambda1 = 800
                        # update y_tilde by back-propagation
                        yy.data.sub_(lambda1 * yy.grad.data)

                        new_y[index, :] = yy.data.cpu().numpy()'''

                    print(epoch)
                    # 每训练1个batch打印一次loss和准确率

                    sum_loss += losses.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    train_correct += predicted.eq(labels_two.data).cpu().sum()  # labels

                    top10_p, top10_in = torch.topk(F.softmax(outputs, dim=1).data, k=10, dim=1)
                    # ======================================================
                    train_loss_n = sum_loss / (i + 1)
                    train_acc_n = (train_correct / total).data.numpy()

                    noise_n = (1 - noise_num / total).data.numpy()
                    noise_pre_n = (noise_jisuan / total).data.cpu().numpy()
                    noise_zip_n = (noise_zide / total).data.cpu().numpy()
                    jiu_true_n = (jiu_num10 / total).data.cpu().numpy()
                    pre_cs_n = (pre_cs / total).data.cpu().numpy()
                    pre_lab_cs_n = (pre_lab_cs / total).data.cpu().numpy()
                    # ==============================================
                    dimt_noise += dimt
                    print(
                        '[epoch:%d, iter:%d] Loss: %.05f | Acc: %.3f%% | noise: %.3f%% | noise_pre: %.3f%% | noise_zp: %.3f%% | jiu_true: %.3f%% | jiu_10: %.3f%% | jiu_9: %.3f%% | jiu_8: %.3f%% | pre_max: %.3f%% | pre_max_max: %.3f%% '
                        % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * train_correct / total,
                           100. * (1 - noise_num / total), 100. * noise_jisuan / total,
                           100. * noise_zide / total, 100. * jiu_num / total, 100. * jiu_num10 / total,
                           100. * jiu_num9 / total, 100. * jiu_num8 / total, 100. * pre_max_sum / total,
                           100. * pre_max_max / total))
                    f2.write(
                        '%03d  %05d |Loss: %.03f | Acc: %.3f%% | noise: %.3f%% | noise_pre: %.3f%% | noise_zp: %.3f%% | jiu_true: %.3f%% | jiu_10: %.3f%% | jiu_9: %.3f%% | jiu_8: %.3f%% | last4: %.3f%% | pre_max: %.3f%% | pre_max_max: %.3f%% | pre_cs: %.3f%% | pre_lab_cs: %.3f%% | tk: %.3f%% | tk2: %.3f%% | dimt: %.3f%%'
                        % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * train_correct / total,
                           100. * (1 - noise_num / total), 100. * noise_jisuan / total,
                           100. * noise_zide / total, 100. * jiu_num / total, 100. * jiu_num10 / total,
                           100. * jiu_num9 / total, 100. * jiu_num8 / total, 100. * last4 / total,
                           100.0 * pre_max_sum / total, 100. * pre_max_max / total, 100. * pre_cs / total,
                           100. * pre_lab_cs / total, 1.00 / r3_min.mean(),
                           (1 - (label_red3[index_t] * pred_s).sum(dim=1)).mean(), 100. * dimt_noise / total))
                    f2.write('\n')
                    f2.flush()


                lp = sum_loss / (i + 1) * ltr
                ltr *= 1.001
                print('lp: ', lp)
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    test_correct = 0
                    test_total = 0
                    test_loss = 0.0
                    for data in testloader:
                        net.eval()
                        # net_class.eval()
                        images_test, labelts = data
                        images_test, labelts = images_test.to(device), labelts.to(device)
                        test_outputs,test_x,test_conf = net(images_test)  # the last one

                        # outputs = net_class(output)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, test_predicted = torch.max(test_outputs.data, 1)

                        labelsn = labelts.data.cpu().numpy()
                        predicteds = test_predicted.cpu().numpy()
                        print('标签:', labelsn)
                        print('判错:', (test_predicted != labelts).data.cpu().to(torch.int32).numpy())
                        print('错标:', (test_predicted != labelts).data.cpu().numpy() * labelsn)
                        print('错预:', (test_predicted != labelts).data.cpu().numpy() * predicteds)

                        for i in range(0, len((test_predicted != labelts).data.cpu().numpy())):
                            if (test_predicted != labelts).data.cpu().to(torch.int32).numpy()[i] == 1:
                                # print('array::::', arry)
                                arry[(test_predicted != labelts).data.cpu().to(torch.int).numpy()[i] * labelsn[i]] += 1

                        # print('arry:', arry)
                        test_loss += criterion(test_outputs, labelts)
                        test_total += labelts.size(0)
                        test_correct += (test_predicted == labelts).sum()
                    print('测试分类准确率为：%.3f%%' % (100. * test_correct / test_total))  # 在100后面加．／／／
                    acc = 100. * test_correct / test_total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc

                        # ========================================save_statistic===========================================================
                        # if continue_from_epoch != -1:        #if this is a new experiment
                        # statistics file
                    test_loss_n = (test_loss / test_total).data.cpu().numpy()
                    test_acc_n = (test_correct / test_total).data.cpu().numpy()

                save_csv(logs_filepath, experiment_name,
                         [epoch + 1, train_loss_n, train_acc_n,
                          noise_n, noise_pre_n, noise_zip_n,
                          jiu_true_n, pre_cs_n, pre_lab_cs_n,
                          "val_loss", "val_acc",
                          test_loss_n, test_acc_n])

                # ==============================================================================
            print("Training Finished, TotalEPOCH=%d" % EPOCH)