# 从 visualization 模块中导入 plot_predictions 和 plot_training_history 函数
from .visualization import plot_predictions, plot_training_history
# plot_predictions 定义在 src/util/visualization.py 中，用于绘制模型的预测结果。
# plot_training_history 定义在 src/util/visualization.py 中，用于绘制训练和验证过程中的损失和准确率曲线。

# 从 model 模块中导入 summarize_model 函数
from .model import summarize_model
# summarize_model 定义在 src/util/model.py 中，用于总结并打印模型的详细信息，包括每一层的名称、参数的数量和参数的形状。

# 模块回顾:
# 1. visualization 模块:
# - plot_predictions: 用于绘制模型的预测结果。定义在 src/util/visualization.py 中。
#   - 功能: 该函数将模型设置为评估模式，使用验证数据加载器生成预测结果，并将结果与原始图像一起显示。具体来说，它会在 10x10 的子图网格中展示模型的前 100 张预测图像及其对应的预测标签。
#   - 主要步骤:
#     1. 将模型设置为评估模式。
#     2. 使用数据加载器批量获取输入数据和标签。
#     3. 对输入数据进行模型预测，并收集图像、真实标签和预测标签。
#     4. 创建 10x10 的子图网格，并显示每张图像及其预测结果。

# - plot_training_history: 用于绘制训练和验证过程中的损失和准确率曲线。定义在 src/util/visualization.py 中。
#   - 功能: 该函数绘制训练过程中的损失和准确率变化曲线，以便可视化模型在训练和验证集上的表现。它使用 matplotlib 创建包含两个子图的绘图窗口，一个显示损失曲线，另一个显示准确率曲线。
#   - 主要步骤:
#     1. 创建一个包含两个子图的绘图窗口。
#     2. 在第一个子图上绘制训练和验证损失曲线。
#     3. 在第二个子图上绘制训练和验证准确率曲线。
#     4. 设置子图标题、轴标签和图例。
#     5. 调整子图间距并显示绘图结果。

# 2. model 模块:
# - summarize_model: 用于总结并打印模型的详细信息。定义在 src/util/model.py 中。
#   - 功能: 该函数输出模型的详细信息，包括每一层的名称、参数的数量、参数的形状以及总参数数量。它遍历模型的所有参数，并打印每一层的详细信息，最后汇总并打印模型的总参数数量。
#   - 主要步骤:
#     1. 打印模型摘要的标题和表头。
#     2. 初始化总参数数量。
#     3. 遍历模型的所有参数，并过滤掉不需要梯度计算的参数。
#     4. 获取每层参数的数量和形状，并累加到总参数数量中。
#     5. 打印每层的详细信息，包括层名称、参数形状和参数数量。
#     6. 打印模型的总参数数量。

# 这些详细注释有助于理解每个模块的功能、具体步骤和在项目中的使用情况，便于后续的维护和使用。
