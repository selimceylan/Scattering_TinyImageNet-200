import os
import numpy as np
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets
import argparse
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Scattering2dResNet(nn.Module):
    def __init__(self, in_channels,  k=2, n=3, num_classes=200):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            nn.Conv2d(in_channels, self.ichannels,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.K, 16, 16)
        x = self.init_conv(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def train(model, device, train_loader, optimizer, epoch, scattering):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(scattering(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def model_test(model, device, test_loader, scattering):
    """
    Attributes:
        model: Created ResNet model.
        device: Specifies to use gpu or cpu.
        test_loader: Test dataloader.
        scattering: Scattering operation.
    Returns:
        correct: Computed accuracy from validation dataset.
        test_loss: Computed loss from validation dataset.
    """

    model.eval()
    # Define variables for loss and accuracy.
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Get scores as "output" from model.
            output = model(scattering(data))
            # Sum up batch loss.
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # Get the index of the max log-probability.
            pred = output.max(1, keepdim=True)[1]
            # Compute accuracy.
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct, test_loss


def model_test_with_train_dataset(model, device, train_loader, scattering):
    """
    Attributes:
        model: Created ResNet model.
        device: Specifies to use gpu or cpu.
        train_loader: Train dataloader.
        scattering: Scattering operation.
    Returns:
        correct: Computed accuracy from train dataset.
        test_loss: Computed loss from train dataset.

    """

    # Define variables for loss and accuracy.
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # Get scores as "output" from model.
            output = model(scattering(data))
            # Sum up batch loss.
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # Get the index of the max log-probability.
            pred = output.max(1, keepdim=True)[1]
            # Compute accuracy.
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return correct, test_loss

def save_ckp(state, checkpoint_dir):
    """
    This function created for save checkpoint file.
    Attributes:
        state: Checkpoint informations('epoch', 'state_dict', 'optimizer').
        checkpoint_dir: Path for saving model file.
    Returns:
        correct: Computed accuracy from train dataset.
        test_loss: Computed loss from train dataset.

    """

    torch.save(state, checkpoint_dir)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    This function created for load checkpoint file and continue
    training.
    Attributes:
        checkpoint_fpath: Pretrained model path.
        model: Pretrained  model
        optimizer: Optimizer which used in pretrained model (SGD).
    Returns:
        model: Pretrained  model.
        optimizer: Optimizer which has feature in pretrained model (SGD).
        checkpoint['epoch']: Epoch value in pretrained.
    """

    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
if __name__ == '__main__':

    """Train a simple Hybrid Resnet Scattering + CNN model on Tiny ImageNet-200.
    
    """
    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')
    parser.add_argument('--width', type=int, default=4,help='width factor for resnet')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.mode == 1:
        scattering = Scattering2D(J=2, shape=(64, 64), max_order=1)
        K = 17*3
    else:
        scattering = Scattering2D(J=2, shape=(32, 32))
        K = 81*3
    if use_cuda:
        scattering = scattering.cuda()




    model = Scattering2dResNet(K, args.width).to(device)
    summary(model,(3,17,16,16))

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
        pin_memory = False

    ROOT = 'Scattering/data-tiny'
    data_dir = os.path.join(ROOT, 'tiny-imagenet-200')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val_prepared')
    means = [0.4802, 0.4481, 0.3975]
    stds = [0.2296, 0.2263, 0.2255]

    train_transforms = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=means,
                             std=stds)
    ])
    # Convert to tensor and normalize validation dataset.
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means,
                             std=stds)
    ])

    # Apply data transforms.
    train_data = datasets.ImageFolder(root="../Scattering/data-tiny/tiny-imagenet-200/train",
                                      transform=train_transforms)
    val_data = datasets.ImageFolder(root="../Scattering/data-tiny/tiny-imagenet-200/val_prepared",
                                    transform=val_transforms)

    BATCH_SIZE = 128
    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory)

    val_iterator = data.DataLoader(val_data,
                                   batch_size=BATCH_SIZE,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory
                                   )

    epoch_value = 15
    correct = np.zeros(epoch_value)
    test_loss = np.zeros(epoch_value)
    correct_intrain = np.zeros(epoch_value)
    test_loss_intrain = np.zeros(epoch_value)
    SAVE_PATH = "../Scattering/Model_tiny_15epoch.pth"
    # Optimizer
    lr = 0.1
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
    #                                     weight_decay=0.0005)
    # ckp_path = "../Scattering/Model_tiny_15epoch_default.pth"
    # model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
    for epoch in range(0, epoch_value):
        if epoch % 4 == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr *= 0.2


        train(model, device, train_iterator, optimizer, epoch + 1, scattering)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        save_ckp(checkpoint, SAVE_PATH)
        correct[epoch], test_loss[epoch] = model_test(model, device, val_iterator, scattering)
        correct_intrain[epoch], test_loss_intrain[epoch] = model_test_with_train_dataset(model, device,
                                                                                             train_iterator, scattering)

    # Create plots for loss visualization.
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(test_loss, 'b', label='loss in test dataset', linewidth=2)
    ax.plot(test_loss_intrain, 'r', label='loss in train dataset', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()


    # Create plots for accuracy visualization.
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot((correct // 100), 'b', label='accuracy in test dataset', linewidth=2)
    ax.plot((correct_intrain // 1000), 'r', label='accuracy in train dataset', linewidth=2)
    ax.set_title('Model accuracy', fontsize=16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()
