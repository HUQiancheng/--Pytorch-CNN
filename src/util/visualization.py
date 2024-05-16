import matplotlib.pyplot as plt  # 引入 matplotlib 库用于绘图
import numpy as np  # 引入 NumPy 库用于数值计算
import torch  # 引入 PyTorch 库

def imshow(img, ax):
    """
    显示图像的辅助函数

    参数:
    img (torch.Tensor): 图像张量
    ax (matplotlib.axes.Axes): matplotlib 子图对象
    """
    img = img / 2 + 0.5  # 反标准化，将图像数据从 [-1, 1] 还原到 [0, 1]
    npimg = img.numpy()  # 将图像张量转换为 NumPy 数组
    # 转置图像数组的维度，将 (C, H, W) 转为 (H, W, C) 并显示图像
    ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

def show_sample(dataset, index):
    """
    显示数据集中某一索引处的图像和标签

    参数:
    dataset (torch.utils.data.Dataset): 数据集对象
    index (int): 图像和标签的索引
    """
    image, label = dataset[index]  # 获取指定索引处的图像和标签
    fig, ax = plt.subplots()  # 创建一个子图
    imshow(image, ax)  # 显示图像
    plt.title(f'Label: {label}')  # 设置标题为标签值
    plt.show()  # 显示子图

def plot_predictions(model, dataloader, device):
    """
    绘制模型的预测结果

    参数:
    model (torch.nn.Module): 训练好的模型
    dataloader (torch.utils.data.DataLoader): 数据加载器
    device (torch.device): 设备（CPU 或 GPU）
    """
    model.eval()  # 将模型设置为评估模式
    images, labels, predictions = [], [], []  # 初始化图像、标签和预测列表
    with torch.no_grad():  # 在不计算梯度的上下文中进行预测
        for i, data in enumerate(dataloader):  # 遍历数据加载器中的数据
            inputs, labels_batch = data  # 获取输入数据和对应的标签
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)  # 将数据和标签移动到指定的设备上（CPU 或 GPU）
            outputs = model(inputs)  # 获取模型输出
            _, predicted_batch = torch.max(outputs, 1)  # 获取预测的类别

            images.extend(inputs.cpu())  # 将输入数据添加到图像列表中，并移动到 CPU
            labels.extend(labels_batch.cpu())  # 将标签添加到标签列表中，并移动到 CPU
            predictions.extend(predicted_batch.cpu())  # 将预测结果添加到预测列表中，并移动到 CPU

            if len(images) >= 100:  # 如果图像数量达到 100 张，停止获取更多数据
                break

    fig, axes = plt.subplots(10, 10, figsize=(15, 15))  # 创建 10x10 的子图网格
    for idx, ax in enumerate(axes.flat):  # 遍历每个子图
        if idx < len(images):  # 如果索引在图像列表范围内
            img = images[idx]  # 获取图像
            img = img / 2 + 0.5  # 反标准化，将图像数据从 [-1, 1] 还原到 [0, 1]
            npimg = img.numpy().squeeze()  # 将图像张量转换为 NumPy 数组并去掉多余的维度
            ax.imshow(npimg, cmap='gray')  # 显示图像
            ax.set_title(f"Prediction: {predictions[idx].item()}", fontsize=8)  # 设置标题为预测结果
            ax.axis('off')  # 关闭坐标轴
    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示绘图结果

def plot_training_history(history):
    """
    绘制训练和验证过程中的损失和准确率曲线

    参数:
    history (dict): 训练过程中记录的损失和准确率
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 创建一个包含两个子图的绘图窗口

    # 绘制训练和验证损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')  # 设置子图标题为 Loss
    ax1.set_xlabel('Epoch')  # 设置 X 轴标签为 Epoch
    ax1.set_ylabel('Loss')  # 设置 Y 轴标签为 Loss
    ax1.legend()  # 显示图例

    # 绘制训练和验证准确率曲线
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Accuracy')  # 设置子图标题为 Accuracy
    ax2.set_xlabel('Epoch')  # 设置 X 轴标签为 Epoch
    ax2.set_ylabel('Accuracy')  # 设置 Y 轴标签为 Accuracy
    ax2.legend()  # 显示图例

    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示绘图结果

# 注释解释:

# 1. 导入库:
# - `matplotlib.pyplot`: 用于绘制图像和图表。
# - `numpy`: 用于数值计算和数组操作。
# - `torch`: PyTorch 的核心库，用于张量操作和模型管理。

# 2. `imshow` 函数:
# - 用于显示图像张量。
# - 参数:
#   - `img`: 图像张量，通常是经过标准化的张量。
#   - `ax`: matplotlib 的子图对象，用于显示图像。
# - 功能:
#   - 反标准化图像张量，将数据从 [-1, 1] 还原到 [0, 1]。
#   - 将图像张量转换为 NumPy 数组，并调整维度顺序以适应 matplotlib 的显示格式。
#   - 在指定的子图对象上显示图像。

# 3. `show_sample` 函数:
# - 用于显示数据集中某一索引处的图像和标签。
# - 参数:
#   - `dataset`: 数据集对象。
#   - `index`: 图像和标签的索引。
# - 功能:
#   - 从数据集中获取指定索引的图像和标签。
#   - 创建一个新的子图对象。
#   - 使用 `imshow` 函数显示图像。
#   - 设置图像标题为标签值。
#   - 显示子图。

# 4. `plot_predictions` 函数:
# - 用于绘制模型的预测结果。
# - 参数:
#   - `model`: 训练好的模型。
#   - `dataloader`: 数据加载器，用于批量加载数据。
#   - `device`: 设备（CPU 或 GPU），用于计算。
# - 功能:
#   - 将模型设置为评估模式。
#   - 初始化图像、标签和预测列表。
#   - 使用 `torch.no_grad` 上下文管理器，在不计算梯度的情况下进行预测。
#   - 遍历数据加载器中的数据，获取输入数据和标签，并移动到指定设备。
#   - 获取模型输出，并提取预测类别。
#   - 将输入数据、标签和预测结果添加到相应的列表中，并移动到 CPU。
#   - 创建一个 10x10 的子图网格。
#   - 在每个子图上显示图像和预测结果。
#   - 调整子图间距并显示绘图结果。

# 5. `plot_training_history` 函数:
# - 用于绘制训练和验证过程中的损失和准确率曲线。
# - 参数:
#   - `history`: 字典，记录了训练过程中的损失和准确率。
# - 功能:
#   - 创建一个包含两个子图的绘图窗口。
#   - 在第一个子图上绘制训练和验证损失曲线。
#   - 在第二个子图上绘制训练和验证准确率曲线。
#   - 设置子图标题、轴标签和图例。
#   - 调整子图间距并显示绘图结果。
