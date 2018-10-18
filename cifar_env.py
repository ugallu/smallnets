import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pylab as pl
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import datetime

# runs training and shows result on cifar10 data
def run_cifar_env(net):

    # normalize
    normalize_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    #trainset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download = True, transform=normalize_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle=True, num_workers = 2)
    #testset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=normalize_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))

    # get random images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # img on grid
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)


    def show_loss(x1):
        pl.clf()
        x2 = np.convolve(x1,np.ones(20,dtype=float),'valid')
        x2 *= 0.05
        pl.plot(x1[:])
        pl.plot(x2[:])
        display.clear_output()
        display.display(pl.gcf())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    # print(device)
    net.to(device)

    started_at = time.time()

    print("Started training")
    loss_history = []
    epoch_history = []
    for epoch in range(10):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            epoch_history.append(time.time() - started_at)
            running_loss += loss.item()
            
            if i % 1000 == 1:
                print('[%d %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0

    training_time = (time.time() - started_at)
    print("Finished training, it took %.3f secound" % training_time)

    print("test the results")

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    images, labels = images.to(device), labels.to(device)

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))


    print("Test on full dataset")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    print("Detailed review of results")
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    return (str(100 * correct / total), str(training_time))
    