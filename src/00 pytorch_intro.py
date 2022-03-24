from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

np.random.seed(42)
torch.manual_seed(42)
plt.style.use("seaborn-white")


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 6 * 6, out_features=120)
        self.fc2 = nn.Linear(120, out_features=84)
        self.fc3 = nn.Linear(84, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(input=F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(input=F.relu(self.conv1(x)), kernel_size=(2,))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


x = torch.randn(size=(100, 1)) * 10
y = 3 * x + 5 * torch.randn(size=(100, 1))

# plt.plot(x.numpy(), y.numpy(), "o")
# plt.grid()
# plt.show()


class LinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        pred = self.linear(x)
        return pred


model = LinearRegression()
print(model)
print(list(model.parameters()))

W, b = model.parameters()


def get_params():
    return W[0][0].item(), b[0].item()


w1, b1 = get_params()
x1 = np.array([-30, 30])
y1 = w1 * x1 + b1


loss_fn = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.001)

epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(y, pred)
    losses.append(loss.detach().numpy())
    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch: {epoch:3d}\t loss: {loss:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("initial model")
plt.plot(x1, y1, "b")
plt.scatter(x, y, marker="o")
plt.subplot(1, 3, 2)
plt.plot(range(epochs), losses)

w1, b1 = get_params()
x1 = np.array([-30, 30])
y1 = w1 * x1 + b1
plt.title("fitted")
plt.subplot(1, 3, 3)
plt.plot(x1, y1, color="red")
plt.scatter(x, y, marker="o")
plt.show()
