#!/usr/bin/env python


#####
# simple implementation of "Densely Connected Convolutional Networks, CVPR 2017"
#####

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np

class DenseBlock(nn.Module):
    """ Dense blocks introduced in paper
    """
    def __init__(self, k_0, k, L, use_dropout = False):
        """
        Args:
            k_0 (int): number of input channels to dense block
            k (int): growth rate described in paper
            L (int): number of layers in dense block
            use_dropout (bool) : whether use dropout after conv layer
        """
        super(DenseBlock, self).__init__()
        self.L = L
        self.use_dropout = use_dropout

        for i in range(L):
            # TODO: per-dimension batchnormalization is provided in torch.nn
            #       however, in the original batchnormalization paper, 
            #       per feature map batchnormalization is applied for convolutional layer
            n_in = k_0 if i == 0 else (k_0 + i * k)     # number of input channels
            self.add_module('bn' + str(i), nn.BatchNorm2d(n_in))
            self.add_module('conv' + str(i), nn.Conv2d(n_in, k, 3, padding = 1))

    def forward(self, x):
        children = self.children()
        for i in range(self.L):
            bn = children.next()
            y = F.relu(bn(x))
            conv = children.next()
            y = conv(y)
            if self.use_dropout:
                y = F.dropout(y, p = 0.2, training = self.training)

            if (i + 1) == self.L:
                x = y  # return last conv layer output
            else:
                x = torch.cat((x, y), 1)
        return x

class TransitionLayer(nn.Module):
    """ TransitionLayer between dense blocks
    """
    def __init__(self, n_in, n_out, use_dropout = False):
        """
        Args:
            n_in (int) : number of input channels
            n_out (int) : number of output channels
            use_dropout (bool) : whether use dropout after conv layer
        """
        super(TransitionLayer, self).__init__()

        self.conv1x1 = nn.Conv2d(n_in, n_out, 1)   # 1x1 conv layer
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1x1(x)
        if self.use_dropout:
            x = F.dropout(x, p = 0.2, training = self.training)
        x = F.avg_pool2d(x, 2)
        return x

def init_weights(m):
    """
    TODO: initialization
    """
    pass

class DenseNet(nn.Module):
    """ Whole framework of dense net for 32 x 32 color image
    """
    def __init__(self, k, L, C, use_dropout = False):
        """
        Args:
            k (int): growth rate for denseblocks
            L (int): number of layers for denseblocks
            C (int) : number of classes
            use_dropout (bool) : whether use dropout after conv layer
        """
        super(DenseNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)   # first conv layer applied on input images

        # dense blocks, connected by transition layer
        self.db1 = DenseBlock(16, k, L, use_dropout)
        self.trl1 = TransitionLayer(k, k, use_dropout)
        self.db2 = DenseBlock(k, k, L, use_dropout)
        self.trl2 = TransitionLayer(k, k, use_dropout)
        self.db3 = DenseBlock(k, k, L, use_dropout)

        # linear layer
        self.fc = nn.Linear(k, C)

    def forward(self, x):
        x = self.conv1(x)
        x = self.db1(x)
        x = self.trl1(x)
        x = self.db2(x)
        x = self.trl2(x)
        x = self.db3(x)
        # global average pooling
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]   # all dimension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def split_train(dataset, valid_percentage):
    """
    Utility function to  split training set into training and validation
    return sample index of both training and validation

    Args:
        dataset (Dataset) : original training dataset
        valid_percentage (float) : percentage of validation

    Returns:
        train_idx (list) : training split index
        valid_idx (list) : validataion split index
    """
    n = len(dataset)
    n_valid = int(np.floor(n * valid_percentage))
    perm  = np.random.permutation(n)
    train_idx = perm[:(n - n_valid)]
    valid_idx = perm[(n - n_valid):]
    return (train_idx, valid_idx)

if __name__ == '__main__':
    #######
    # test on CIFAR10 dataset
    #######

    # compute CIFAR10 mean and variance
    #data_dir = '../data/cifar10/'
    data_dir = './cifar10/'
    data = torchvision.datasets.CIFAR10(root = data_dir, train = True, download = True).train_data
    data = data.astype(np.float32) / 255.
    cifar_mean = np.mean(data, axis = (0, 1, 2))
    cifar_std = np.std(data, axis = (0, 1, 2))
    cifar_mean = torch.from_numpy(cifar_mean).float()
    cifar_std = torch.from_numpy(cifar_std).float()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(cifar_mean, cifar_std)])
    augment = None  # no data augmentation
    #augment = transforms.Compose([])   # data augmentation
    valid_transform = transform

    if augment is None:
        train_transform = transform
    else:
        train_transform = transforms.Compose([augment, transform])

    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    kwargs = {'num_workers' : 1, 'pin_memory': True} if use_cuda else {}

    # simply duplicate dataset
    cifar10_train = torchvision.datasets.CIFAR10(root = data_dir, train = True,
                                                 download = True, transform = train_transform)
    train_idx, valid_idx = split_train(cifar10_train, 0.1)   # 5000 validation samples

    cifar10_valid = torchvision.datasets.CIFAR10(root = data_dir, train = True,
                                                 download = True, transform = valid_transform)

    batch_sz = 64
    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size = batch_sz,
            sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx),
            **kwargs)
    valid_loader = torch.utils.data.DataLoader(cifar10_valid, batch_size = batch_sz,
            sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx),
            **kwargs)

    # last epoch use the whole training dataset
    whole_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size = batch_sz, **kwargs)

    #######
    # training stage
    ######

    #net = DenseNet(5, 3, 10, True)     # just for CPU test
    cuda_device = 2     # avoid Tensors on different GPU
    net = DenseNet(12, 25, 10, use_dropout = True)

    if use_cuda:
        net.cuda(cuda_device)

    criterion = nn.CrossEntropyLoss()
    init_lr = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    optimizer = optim.SGD(net.parameters(), lr = init_lr, weight_decay = weight_decay,
            momentum = momentum, nesterov = True)

    n_epoch = 300
    valid_freq = 50  # validation frequency

    net.train()      # training mode for dropout and batch normalization layer
    for epoch in range(n_epoch):
        running_loss = 0.0

        if epoch + 1 == n_epoch:  ## use the whole training in last epoch
            data_loader = whole_train_loader
        else:
            data_loader = train_loader

        # divide lr by 10 after 50% and 75% epochs
        if epoch + 1 == .5 * n_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr * 0.1
        if epoch + 1 == .75 * n_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr * 0.01

        for i, data in enumerate(data_loader, 0):
            inputs, labels = data

            # wrap in variable
            if use_cuda:
                inputs, labels = Variable(inputs.cuda(cuda_device)), Variable(labels.cuda(cuda_device))
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            # print statistics
            if (i + 1) %  valid_freq == 0:
                print('[%d, %d] training loss: %.4f' %(epoch + 1,
                    i + 1, running_loss / valid_freq))
                running_loss = 0.

                # validation loss
                valid_loss_val = 0.0
                net.eval()      # eval mode
                for inputs, labels in valid_loader:
                    if use_cuda:
                        inputs = Variable(inputs.cuda(cuda_device), volatile = True)
                        labels = Variable(labels.cuda(cuda_device))
                    else:
                        inputs, labels = Variable(inputs, volatile = True), Variable(labels)

                    outputs = net(inputs)
                    valid_loss = criterion(outputs, labels)
                    valid_loss_val += valid_loss.data[0]
                print('\t\t validation loss: %.4f' %(valid_loss_val / len(valid_loader)))
                net.train()

    print('Finished Training')

    #######
    # testing stage
    #######
    test_dataset = torchvision.datasets.CIFAR10(root = data_dir, train = False,
            download = True, transform = valid_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_sz,
            shuffle = False, **kwargs)

    net.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        if use_cuda:
            inputs = Variable(inputs.cuda(cuda_device), volatile = True)
            #labels = Variable(labels.cuda(cuda_device))
        else:
            #inputs, labels = Variable(inputs, volatile = True), Variable(labels)
            inputs = Variable(inputs, volatile = True)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).squeeze().sum()
        total += labels.size(0)

    print ("Test accuracy : %f %% " % (correct * 100.0 / total))
