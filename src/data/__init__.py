# 从 CNN_Dataset 模块中导入 CNN_Dataset 类和 transform 函数
from .CNN_Dataset import CNN_Dataset, transform
# CNN_Dataset 定义在 src/data/CNN_Dataset.py 中，用于加载和处理图像数据。
# transform 定义在 src/data/CNN_Dataset.py 中，用于图像预处理和转换操作。

# 模块回顾:
# 1. CNN_Dataset 模块:
# - CNN_Dataset: 自定义数据集类，用于加载和处理图像数据。定义在 src/data/CNN_Dataset.py 中。
#   - 功能: 该类用于从指定的目录结构中加载图像文件，并进行必要的预处理操作，如颜色反转和标准化。它还实现了数据集的划分，用于训练集和验证集的创建。
#   - 主要步骤:
#     1. 在 __init__ 方法中，定义根目录、数据转换操作、是否为训练集标志和训练集比例。
#     2. 使用 _load_data 方法加载图像文件路径和对应的标签。
#     3. 使用 train_test_split 函数将数据集划分为训练集和验证集。
#     4. 在 __getitem__ 方法中，读取指定索引处的图像文件，并进行预处理操作，如转换为灰度图像、反转颜色和标准化。
#     5. 返回处理后的图像和对应的标签。

# - transform: 数据转换操作，用于图像预处理和标准化。定义在 src/data/CNN_Dataset.py 中。
#   - 功能: 该变量定义了一系列图像预处理操作，包括将图像转换为张量和标准化处理，以确保图像数据在训练过程中具有相同的尺度。
#   - 主要步骤:
#     1. 使用 transforms.Compose 函数组合多个图像预处理操作。
#     2. 使用 transforms.ToTensor 将图像转换为 PyTorch 张量。
#     3. 使用 transforms.Normalize 标准化图像数据，使其均值为 0.5，标准差为 0.5。

# 详细注释解释:
# - __init__ 方法:
#   - 参数:
#     - root_dir (str): 数据集的根目录。
#     - transform (callable, optional): 应用于图像的转换函数。
#     - train (bool): 指定是训练集还是验证集。
#     - train_ratio (float): 训练集比例，默认为0.8。
#   - 功能:
#     - 定义根目录、数据转换操作、是否为训练集标志和训练集比例。
#     - 调用 _load_data 方法加载图像文件路径和标签。
#     - 使用 train_test_split 将数据集划分为训练集和验证集。

# - _load_data 方法:
#   - 功能: 加载数据，将图像文件路径和标签存储到相应的列表中。
#   - 步骤:
#     1. 遍历每个标签目录，获取图像文件路径。
#     2. 将图像文件路径和标签存储到对应的列表中。

# - __getitem__ 方法:
#   - 参数:
#     - idx (int): 图像和标签的索引。
#   - 功能:
#     - 读取指定索引处的图像文件，并进行预处理操作。
#   - 步骤:
#     1. 获取图像文件路径。
#     2. 打开图像文件并转换为灰度图像。
#     3. 反转颜色并标准化图像数据。
#     4. 返回处理后的图像和标签。

# - __len__ 方法:
#   - 功能: 返回数据集的大小，即图像的数量。

# 这些详细注释有助于理解 CNN_Dataset 类和 transform 变量的功能、具体步骤和在项目中的使用情况，便于后续的维护和使用。
