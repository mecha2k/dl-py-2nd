from importlib import import_module
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")

np.random.seed(42)
torch.manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5))]
)

trainset = torchvision.datasets.MNIST(
    root="../data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="../data", train=False, download=True, transform=transform
)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)

image, label = next(iter(train_loader))
print(image.shape)
print(label.shape)


def image_show(image):
    image = image / 2 + 0.5
    plt.figure(figsize=(10, 6))
    plt.imshow(np.transpose(image.numpy(), axes=(1, 2, 0)))
    plt.show()


dataset = iter(train_loader)
images, labels = dataset.next()
print(images.shape)
# image_show(torchvision.utils.make_grid(images[0]))


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(input=F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(input=F.relu(self.conv2(x)), kernel_size=2)
        x = x.view(-1, self.num_flat_shape(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_shape(self, x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].shape)


inputs = torch.randn(size=(1, 1, 28, 28))
outputs = net(inputs)
print(outputs)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

total_batch = len(train_loader)
print(total_batch)

epochs = 2

for epoch in range(epochs):
    losses = 0.0
    for i, data in enumerate(train_loader):
        images, labels = data

        print(images[0].shape)
        break

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        if i % 100 == 0:
            print(f"epoch: {epoch:3d}, iter: {i:3d}, loss: {losses / 2000}")
            losses = 0.0


x = torch.randn(size=(2, 28, 28))
y = torch.randn(size=(28, 56))
z = torch.mm(x, y)

print(z.shape)