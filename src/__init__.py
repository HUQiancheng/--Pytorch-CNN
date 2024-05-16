# 导入自定义数据处理模块，包括数据集类和数据转换操作
from .data import CNN_Dataset, transform
# CNN_Dataset 定义在 src/data/CNN_Dataset.py 中，用于加载和处理图像数据。
# transform 是用于图像预处理和转换的操作，通常用于标准化图像数据。

# 导入模型加载模块
from .inference import load_model
# load_model 定义在 src/inference/CNN_Inference.py 中，用于加载已训练好的模型。
# 在主函数中用于检查是否存在已保存的模型，并根据用户选择加载模型。

# 导入模型定义模块
from .model import CNNModel
# CNNModel 定义在 src/model/CNNModel.py 中，包含卷积神经网络模型的结构定义。
# 在主函数中实例化模型，并在训练和预测时使用。

# 导入训练相关模块，包括训练函数和数据加载器创建函数
from .training import train, create_dataloader
# train 定义在 src/training/train_CNN.py 中，用于训练模型。
# create_dataloader 定义在 src/training/train_CNN.py 中，用于创建训练数据加载器和验证数据加载器。

# 导入实用工具模块，包括预测结果绘制、训练历史绘制和模型总结函数
from .util import plot_predictions, plot_training_history, summarize_model
# plot_predictions 定义在 src/util/visualization.py 中，用于绘制模型的预测结果。
# plot_training_history 定义在 src/util/visualization.py 中，用于绘制训练和验证过程中的损失和准确率曲线。
# summarize_model 定义在 src/util/model.py 中，用于总结并打印模型的详细信息，包括每一层的名称、参数的数量和参数的形状。

# 注释解释:
# 1. .data 模块:
# - `CNN_Dataset`: 数据集类，用于加载和处理图像数据。定义在 src/data/CNN_Dataset.py 中。
# - `transform`: 数据转换操作，用于标准化图像数据。定义在 src/data/CNN_Dataset.py 中。

# 2. .inference 模块:
# - `load_model`: 模型加载函数，用于加载已训练好的模型。定义在 src/inference/CNN_Inference.py 中。
# - 在主函数中用于检查是否存在已保存的模型，并根据用户选择加载模型。

# 3. .model 模块:
# - `CNNModel`: 卷积神经网络模型类，包含模型的结构定义。定义在 src/model/CNNModel.py 中。
# - 在主函数中实例化模型，并在训练和预测时使用。

# 4. .training 模块:
# - `train`: 训练函数，用于训练模型。定义在 src/training/train_CNN.py 中。
# - `create_dataloader`: 数据加载器创建函数，用于创建训练数据加载器和验证数据加载器。定义在 src/training/train_CNN.py 中。

# 5. .util 模块:
# - `plot_predictions`: 预测结果绘制函数，用于绘制模型的预测结果。定义在 src/util/visualization.py 中。
# - `plot_training_history`: 训练历史绘制函数，用于绘制训练和验证过程中的损失和准确率曲线。定义在 src/util/visualization.py 中。
# - `summarize_model`: 模型总结函数，用于总结并打印模型的详细信息。定义在 src/util/model.py 中。
