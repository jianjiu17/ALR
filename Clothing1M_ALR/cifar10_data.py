"""
cifar-10 dataset, with support for random labels
"""
import numpy
import numpy as np
from PIL import Image
import torch
import torchvision.datasets as datasets
import random
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

"""label_p =np.load('/home/user_1/zhangwenzhen/resnet-18-cifar10-ce/labcount1/label_pred1.npy')# right
label_p = numpy.array(label_p,dtype=int).reshape([50000])"""

class CIFAR10RandomLabels(datasets.CIFAR10):    #datasets.CIFAR10
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
    #self.transition = {0: 0, 2: 2, 4: 4, 7: 7, 1: 1, 9: 9, 3: 3, 5: 3, 6: 3, 8: 0}
    #self.transition = {0: 4, 2: 2, 4: 4, 7: 3, 1: 1, 9: 6, 3: 3, 5: 3, 6: 6, 8: 2}
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
      noise_mode = 'asym'
      noise_label = []
      idx = list(range(50000))
      random.shuffle(idx)
      num_noise = int(corrupt_prob * 50000)
      noise_idx = idx[:num_noise]
      for i in range(50000):
          if i in noise_idx:
              if noise_mode == 'sym':
                  noiselabel = random.randint(0, 9)
                  noise_label.append(noiselabel)
              elif noise_mode == 'asym':
                  noiselabel = self.transition[self.targets[i]]
                  noise_label.append(noiselabel)
          else:
              noise_label.append(self.targets[i])

      self.targets = noise_label      #self.trainset


class CIFAR10RandomLabels_pred(datasets.CIFAR10):  # datasets.CIFAR10
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 10. The number of classes in the dataset.
    """

    def __init__(self, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels_pred, self).__init__(**kwargs)
        self.n_classes = num_classes
        self.target_c = self.targets
        self.targets = label_p.tolist()

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

        return img, target, target_c, index

