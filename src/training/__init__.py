# 从 train_CNN 模块中导入 train 和 create_dataloader 函数
from .train_CNN import train, create_dataloader
# train 定义在 src/training/train_CNN.py 中，用于训练模型。
# create_dataloader 定义在 src/training/train_CNN.py 中，用于创建训练数据加载器和验证数据加载器。

# 模块回顾:
# 1. train_CNN 模块:
# - train: 用于训练模型。定义在 src/training/train_CNN.py 中。
#   - 功能: 该函数负责训练模型，包括训练和验证两个阶段，并记录训练和验证的损失及准确率。它接收模型、数据加载器、损失函数、优化器、设备和训练轮数等参数。
#   - 主要步骤:
#     1. 初始化记录训练和验证损失、准确率的字典。
#     2. 进入训练循环，对每个 epoch 进行训练和验证。
#     3. 在训练阶段，将模型设置为训练模式，并对每个 batch 执行前向传播、计算损失、反向传播和更新模型参数。
#     4. 在验证阶段，将模型设置为评估模式，并对每个 batch 进行前向传播和计算损失。
#     5. 记录每个 epoch 的训练和验证损失及准确率，并打印结果。
#     6. 返回训练好的模型和训练记录（损失和准确率）。

# - create_dataloader: 用于创建训练数据加载器和验证数据加载器。定义在 src/training/train_CNN.py 中。
#   - 功能: 该函数根据数据集路径、数据转换操作、批量大小和训练集比例等参数，创建并返回用于批量加载数据的 DataLoader。
#   - 主要步骤:
#     1. 实例化自定义数据集 CNN_Dataset，传入根目录、数据转换操作和训练集比例。
#     2. 根据是否为训练集，划分训练集和验证集，并实例化相应的 DataLoader。
#     3. 返回创建好的 DataLoader，用于批量加载训练数据或验证数据。

# 这些详细注释有助于理解每个模块的功能、具体步骤和在项目中的使用情况，便于后续的维护和使用。
