"""
cifar-10 dataset, with support for random labels
"""
import numpy as np
from PIL import Image
import torch
import torchvision.datasets as datasets
import random
import os
import torchvision
#torchvision.datasets.CIFAR100(root='./data1', train=True, download=True) #训练数据集
'''class CIFAR10RandomLabels(datasets.CIFAR10):    #datasets.CIFAR10
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.target_c = self.targets
    self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def __getitem__(self, index: int):
      """
      Args:
          index (int): Index

      Returns:
          tuple: (image, target) where target is index of the target class.
      """
      img, target, target_c = self.data[index], self.targets[index], self.target_c[index]

      # doing this so that it is consistent with all other datasets
      # to return a PIL Image
      img = Image.fromarray(img)

      if self.transform is not None:
          img = self.transform(img)

      if self.target_transform is not None:
          target = self.target_transform(target)

      return img, target, target_c, index
  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)         #self.targets
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    print('sdf: ',mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    self.targets = labels      #self.trainset'''
def train_val_split(base_dataset: torchvision.datasets.CIFAR100):
    num_classes = 100
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

clas = 100
class CIFAR10RandomLabels(datasets.CIFAR100):    #datasets.CIFAR10
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, noise_t = 'sym', num_classes=clas, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.target_c = self.targets
    self.noise_t = noise_t
    self.transition_cifar100 = {}
    base = [1, 2, 3, 4, 0]
    for i in range(100):
        self.transition_cifar100[i] = int(base[i % 5] + 5 * int(i / 5))
    self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
    #self.transition = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def __getitem__(self, index: int):
      """
      Args:
          index (int): Index

      Returns:
          tuple: (image, target) where target is index of the target class.
      """
      img, target, target_c = self.data[index], self.targets[index], self.target_c[index]

      # doing this so that it is consistent with all other datasets
      # to return a PIL Image
      img = Image.fromarray(img)

      if self.transform is not None:
          img = self.transform(img)

      if self.target_transform is not None:
          target = self.target_transform(target)

      return img, target, target_c, index
  def corrupt_labels(self, corrupt_prob):
      noise_mode = self.noise_t
      cifar_num = '100'
      noise_label = []
      idx = list(range(50000))
      random.shuffle(idx)
      num_noise = int(corrupt_prob * 50000)
      noise_idx = idx[:num_noise]
      for i in range(50000):
          if i in noise_idx:
              if noise_mode == 'sym':
                  noiselabel = random.randint(0, clas-1)
                  noise_label.append(noiselabel)
              elif noise_mode == 'asym':
                  if cifar_num == '10':
                      noiselabel = self.transition[self.targets[i]]
                  elif cifar_num == '100':
                      noiselabel = self.transition_cifar100[self.targets[i]]

                  noise_label.append(noiselabel)
          else:
              noise_label.append(self.targets[i])

      self.targets = noise_label      #self.trainset
