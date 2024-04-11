from torch import optim
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os

train_batch_size = 128
test_batch_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mnist_dataset(train):  # 准备minist的dataset
    func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )]
    )
    # 1. 准备Mnist数据集
    return MNIST(root="./data", train=train, download=True, transform=func)


def get_dataloader(train=True):
    mnist = mnist_dataset(train)
    return DataLoader(mnist, batch_size=train_batch_size, shuffle=True)


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, image):
        image_viwed = image.view(-1, 1 * 28 * 28)  # [batch_size,1*28*28]
        fc1_out = self.fc1(image_viwed)  # [batch_size,100]
        fc1_out_relu = F.relu(fc1_out)  # [batch_siz3,100]
        out = self.fc2(fc1_out_relu)  # [batch_size,10]
        return F.log_softmax(out, dim=-1)


# 1. 实例化模型，优化器，损失函数
model = MnistModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# 2. 进行循环，进行训练
def train(epoch):
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    total_loss = []
    for idx, (input, target) in bar:
        input = input.to(device)
        target = target.to(device)
        # 梯度置为0
        optimizer.zero_grad()
        # 计算得到预测值
        output = model(input)
        # 得到损失
        loss = F.nll_loss(output, target)
        # 反向传播，计算损失
        loss.backward()
        total_loss.append(loss.item())
        # 参数的更新
        optimizer.step()
        # 打印数据
        if idx % 10 == 0:
            bar.set_description("epcoh:{} idx:{},loss:{:.6f}".format(epoch, idx, np.mean(total_loss)))
            torch.save(model.state_dict(), "model.pkl")
            torch.save(optimizer.state_dict(), "optimizer.pkl")


def eval():
    # 1. 实例化模型，优化器，损失函数
    model = MnistModel().to(device)
    if os.path.exists("model.pkl"):
        model.load_state_dict(torch.load("model.pkl"))
    test_dataloader = get_dataloader(train=False)
    total_loss = []
    total_acc = []
    with torch.no_grad():
        for input, target in test_dataloader:  # 2. 进行循环，进行训练
            input = input.to(device)
            target = target.to(device)
            # 计算得到预测值
            output = model(input)
            # 得到损失
            loss = F.nll_loss(output, target)
            # 反向传播，计算损失
            total_loss.append(loss.item())
            # 计算准确率
            ###计算预测值
            pred = output.max(dim=-1)[-1]
            total_acc.append(pred.eq(target).float().mean().item())
    print("test loss:{},test acc:{}".format(np.mean(total_loss), np.mean(total_acc)))


if __name__ == '__main__':
    for i in range(10):
        train(i)
        eval()