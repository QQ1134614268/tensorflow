import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 加载测试图像

# 1. 数据预处理  
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


model = Net()

# 3. 定义损失函数和优化器  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个mini-batches打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 5. 测试模型  
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


# 6. 使用模型进行预测（可选）
def predict(image, model, transform):
    # 对图像进行预处理
    image = transform(image).unsqueeze(0).to(device)  # 添加batch维度，并移动到GPU（如果可用）
    # 通过模型进行前向传播
    output = model(image)
    # 获取预测的类别
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()


# 假设您有一个名为'test_image.png'的图像文件，并且它位于当前工作目录中
test_image_path = 'test_image.png'
test_image = Image.open(test_image_path).convert('L')  # 转换为灰度图像，因为MNIST是灰度图像

# 使用模型进行预测
predicted_label = predict(test_image, model, transform)

# 打印预测结果
print(f'Predicted label for the image {test_image_path}: {predicted_label}')

# 7. 保存模型
torch.save(model.state_dict(), 'mnist_model.pth')
print('Model saved successfully!')


# 8. 加载模型进行预测（可选，通常在实际使用时执行此步骤）
# 首先，确保您有一个训练好的模型文件 'mnist_model.pth'
def load_model(model_path):
    model = Net()  # 实例化与之前相同的模型结构
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 将模型设置为评估模式
    return model


# 加载模型
loaded_model = load_model('mnist_model.pth')
loaded_model = loaded_model.to(device)  # 确保模型在正确的设备上


# 使用加载的模型进行预测
def predict_with_loaded_model(image, model, transform):
    # ... [省略predict函数的代码，与之前的predict函数相同]
    # 进行预测
    test_image_path = 'test_image.png'
    test_image = Image.open(test_image_path).convert('L')  # 确保是灰度图像
    predicted_label = predict_with_loaded_model(test_image, loaded_model, transform)
    print(f'Predicted label for the loaded model on image {test_image_path}: {predicted_label}')
