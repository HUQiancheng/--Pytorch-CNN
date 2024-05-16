import torch.nn as nn  # 引入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 引入 PyTorch 的功能性模块，包含常用的激活函数等

# 定义卷积神经网络模型类，继承自 nn.Module
class CNNModel(nn.Module):
    def __init__(self):
        """
        初始化模型结构
        """
        super(CNNModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 定义第一个卷积层，输入通道为1，输出通道为32，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 定义第二个卷积层，输入通道为32，输出通道为64，卷积核大小为3x3，填充为1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 定义最大池化层，池化窗口大小为2x2，步幅为2，填充为0
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 定义第一个全连接层，输入单元数为64*7*7，输出单元数为128
        self.fc2 = nn.Linear(128, 10)  # 定义第二个全连接层，输入单元数为128，输出单元数为10（对应分类数量）

    def forward(self, x):
        """
        定义前向传播过程
        """
        # 输入张量 x 的形状应为 (batch_size, 1, 28, 28)
        # 其中 batch_size 是批次大小，1 表示单通道灰度图，28x28 是图像的空间维度
        x = self.pool(F.relu(self.conv1(x)))  # 第一个卷积层 -> ReLU 激活函数 -> 最大池化层
        x = self.pool(F.relu(self.conv2(x)))  # 第二个卷积层 -> ReLU 激活函数 -> 最大池化层
        x = x.view(-1, 64 * 7 * 7)  # 将张量展平成一维向量，-1 表示自动计算 batch size
        x = F.relu(self.fc1(x))  # 第一个全连接层 -> ReLU 激活函数
        x = self.fc2(x)  # 第二个全连接层，输出最终的分类结果
        return x  # 返回输出

# 注释解释:
# 1. `self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)`: 
#    - 定义了第一个卷积层，输入通道为1（灰度图），输出通道为32，卷积核大小为3x3，填充为1。
# 2. `self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)`:
#    - 定义了第二个卷积层，输入通道为32，输出通道为64，卷积核大小为3x3，填充为1。
# 3. `self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)`:
#    - 定义了一个最大池化层，池化窗口大小为2x2，步幅为2，填充为0。
# 4. `self.fc1 = nn.Linear(64 * 7 * 7, 128)`:
#    - 定义了第一个全连接层，输入单元数为64*7*7，输出单元数为128。
# 5. `self.fc2 = nn.Linear(128, 10)`:
#    - 定义了第二个全连接层，输入单元数为128，输出单元数为10，表示10个分类。

# 前向传播解释:
# 1. `x = self.pool(F.relu(self.conv1(x)))`:
#    - 输入形状 (batch_size, 1, 28, 28) -> 通过第一个卷积层后形状为 (batch_size, 32, 28, 28)
#    - 经过 ReLU 激活函数 -> 最大池化后形状为 (batch_size, 32, 14, 14)
# 2. `x = self.pool(F.relu(self.conv2(x)))`:
#    - 输入形状 (batch_size, 32, 14, 14) -> 通过第二个卷积层后形状为 (batch_size, 64, 14, 14)
#    - 经过 ReLU 激活函数 -> 最大池化后形状为 (batch_size, 64, 7, 7)
# 3. `x = x.view(-1, 64 * 7 * 7)`:
#    - 将张量展平成一维向量，形状为 (batch_size, 64*7*7)
# 4. `x = F.relu(self.fc1(x))`:
#    - 经过第一个全连接层，形状为 (batch_size, 128)
# 5. `x = self.fc2(x)`:
#    - 经过第二个全连接层，形状为 (batch_size, 10)
# 6. `return x`:
#    - 返回输出，形状为 (batch_size, 10)
