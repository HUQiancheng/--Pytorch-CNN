import torch  # 引入 PyTorch 库，用于张量操作和模型管理
import torch.optim as optim  # 引入 PyTorch 中的优化器模块，用于优化模型参数
import torch.nn as nn  # 引入 PyTorch 中的神经网络模块，用于定义损失函数和模型结构
from torch.utils.data import DataLoader  # 引入 PyTorch 中的数据加载器模块，用于批量加载数据
from src.model.CNNModel import CNNModel  # 从自定义模型文件中引入 CNNModel 类，用于构建卷积神经网络模型
from src.data.CNN_Dataset import CNN_Dataset, transform  # 从自定义数据文件中引入 CNN_Dataset 类和 transform 变量，用于数据预处理

def train(model, trainloader, valloader, criterion, optimizer, device, epochs=5, print_interval=100):
    """
    训练模型的函数

    参数:
    model: 训练的模型
    trainloader: 训练数据的数据加载器
    valloader: 验证数据的数据加载器
    criterion: 损失函数
    optimizer: 优化器
    device: 训练使用的设备（CPU 或 GPU）
    epochs (int, optional): 训练的轮数，默认值为 5
    print_interval (int, optional): 每隔多少个 batch 打印一次，默认值为 100

    返回:
    tuple: 训练好的模型，训练过程中的损失和准确率记录
    """
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}  # 初始化记录训练和验证损失、准确率的字典

    for epoch in range(epochs):  # 遍历每一个 epoch
        running_loss = 0.0  # 初始化当前轮次的累计损失
        correct = 0  # 初始化当前轮次的正确预测数
        total = 0  # 初始化当前轮次的总样本数

        # 训练阶段
        model.train()  # 将模型设置为训练模式
        for i, data in enumerate(trainloader, 0):  # 遍历每一个 batch
            inputs, labels = data  # 获取输入数据和对应的标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移动到指定的设备上（CPU 或 GPU）
            optimizer.zero_grad()  # 将优化器的梯度缓存清零
            outputs = model(inputs)  # 前向传播计算模型输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失值
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测数

            if (i + 1) % print_interval == 0:  # 每 print_interval 个 batch 打印一次
                print(f"Batch [{i + 1}/{len(trainloader)}] "
                      f"Running loss: {running_loss / (i + 1):.3f}, accuracy: {100 * correct / total:.2f}%")

        epoch_loss = running_loss / len(trainloader)  # 计算当前轮次的平均损失
        epoch_accuracy = 100 * correct / total  # 计算当前轮次的准确率
        history['train_loss'].append(epoch_loss)  # 记录当前轮次的训练损失
        history['train_accuracy'].append(epoch_accuracy)  # 记录当前轮次的训练准确率

        # 验证阶段
        model.eval()  # 将模型设置为评估模式（测试模式）
        val_loss = 0.0  # 初始化当前轮次的验证损失
        val_correct = 0  # 初始化当前轮次的验证正确预测数
        val_total = 0  # 初始化当前轮次的验证总样本数
        with torch.no_grad():  # 在不计算梯度的上下文中进行验证
            for i, data in enumerate(valloader, 0):  # 遍历每一个 batch
                inputs, labels = data  # 获取输入数据和对应的标签
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移动到指定的设备上（CPU 或 GPU）
                outputs = model(inputs)  # 前向传播计算模型输出
                loss = criterion(outputs, labels)  # 计算损失
                val_loss += loss.item()  # 累加验证损失值
                _, predicted = torch.max(outputs, 1)  # 获取预测的类别
                val_total += labels.size(0)  # 累加验证总样本数
                val_correct += (predicted == labels).sum().item()  # 累加验证正确预测数

        epoch_val_loss = val_loss / len(valloader)  # 计算当前轮次的平均验证损失
        epoch_val_accuracy = 100 * val_correct / val_total  # 计算当前轮次的验证准确率
        history['val_loss'].append(epoch_val_loss)  # 记录当前轮次的验证损失
        history['val_accuracy'].append(epoch_val_accuracy)  # 记录当前轮次的验证准确率

        # 打印当前轮次的训练和验证损失及准确率
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train loss: {epoch_loss:.3f}, accuracy: {epoch_accuracy:.2f}% "
              f"Val loss: {epoch_val_loss:.3f}, accuracy: {epoch_val_accuracy:.2f}%")

    print("Finished Training")  # 打印训练完成信息
    return model, history  # 返回训练好的模型和训练记录

def create_dataloader(root_dir, transform, batch_size=64, train=True, train_ratio=0.8):
    """
    创建数据加载器的函数

    参数:
    root_dir (str): 数据集的根目录
    transform: 图像预处理和转换操作
    batch_size (int, optional): 每个 batch 的样本数量，默认为 64
    train (bool, optional): 指定是训练集还是验证集，默认为 True
    train_ratio (float, optional): 训练集的比例，默认为 0.8

    返回:
    DataLoader: 数据加载器
    """
    dataset = CNN_Dataset(root_dir=root_dir, transform=transform, train=train, train_ratio=train_ratio)  # 实例化自定义数据集
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建数据加载器，并设置 batch size 和是否打乱数据

# 注释解释:

# 1. 导入库:
# - `torch`: PyTorch 的核心库，用于张量操作和模型管理。
# - `torch.optim`: PyTorch 的优化器模块，用于定义和调整模型参数的优化算法。
# - `torch.nn`: PyTorch 的神经网络模块，用于定义损失函数和神经网络层。
# - `torch.utils.data.DataLoader`: 用于创建数据加载器，方便进行批量数据加载。
# - `CNNModel`: 从自定义的模型文件中引入 CNN 模型类，用于定义和构建卷积神经网络。
# - `CNN_Dataset`, `transform`: 从自定义数据文件中引入数据集类和图像转换操作，用于数据预处理和加载。

# 2. `train` 函数:
# - 用于训练模型，包含训练和验证两个阶段，并记录训练和验证的损失及准确率。
# - 参数:
#   - `model`: 要训练的模型。
#   - `trainloader`: 训练数据的数据加载器。
#   - `valloader`: 验证数据的数据加载器。
#   - `criterion`: 损失函数，用于计算模型输出与真实标签之间的差异。
#   - `optimizer`: 优化器，用于更新模型参数。
#   - `device`: 训练使用的设备（CPU 或 GPU）。
#   - `epochs`: 训练的轮数，默认值为 5。
#   - `print_interval`: 每隔多少个 batch 打印一次训练信息，默认值为 100。
# - 返回值: 训练好的模型和记录训练过程中的损失和准确率的字典。

# 3. `create_dataloader` 函数:
# - 用于创建数据加载器，根据提供的数据集路径和图像转换操作实例化数据集，并返回 DataLoader。
# - 参数:
#   - `root_dir`: 数据集的根目录。
#   - `transform`: 图像预处理和转换操作。
#   - `batch_size`: 每个 batch 的样本数量，默认为 64。
#   - `train`: 指定是训练集还是验证集，默认为 True。
#   - `train_ratio`: 训练集的比例，默认为 0.8。
# - 返回值: 数据加载器，用于批量加载数据。

# 详细注释解释:
# - `history` 字典用于记录每个 epoch 的训练和验证损失及准确率。
# - `model.train()` 将模型设置为训练模式，使得 dropout 和 batch normalization 层正常工作。
# - `optimizer.zero_grad()` 清除优化器中的梯度缓存，避免梯度累加。
# - `loss.backward()` 进行反向传播，计算梯度。
# - `optimizer.step()` 更新模型参数。
# - `torch.no_grad()` 上下文管理器，确保在验证阶段不计算梯度，节省内存和计算资源。
# - `model.eval()` 将模型设置为评估模式，冻结 dropout 和 batch normalization 层。
