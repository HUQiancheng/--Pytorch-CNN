# 从 CNN_Inference 模块中导入 load_model 函数
from .CNN_Inference import load_model
# load_model 定义在 src/inference/CNN_Inference.py 中，用于加载已训练好的模型。

# 模块回顾:
# 1. CNN_Inference 模块:
# - load_model: 用于加载已训练好的模型。定义在 src/inference/CNN_Inference.py 中。
#   - 功能: 该函数负责从指定路径加载模型的状态字典，并将其应用到模型实例中。
#   - 主要步骤:
#     1. 实例化 CNNModel 类，创建模型对象。
#     2. 使用 torch.load 函数从指定路径加载模型的状态字典。
#     3. 使用 load_state_dict 方法将加载的状态字典应用到模型实例中。
#     4. 将模型设置为评估模式（eval），以确保在预测时禁用 dropout 和 batch normalization。
#     5. 返回加载好的模型实例。

# 详细注释解释:
# - load_model 函数:
#   - 参数:
#     - model_path (str): 保存模型状态字典的路径。
#   - 返回值:
#     - model (nn.Module): 加载好的模型实例。
#   - 功能:
#     - 从指定路径加载模型的状态字典，并将其应用到 CNNModel 实例中。
#     - 将模型设置为评估模式，以确保在预测时禁用 dropout 和 batch normalization。
#     - 返回加载好的模型实例，用于后续的预测任务。

# - 在主函数中的使用:
#   - 主函数中检查是否存在已保存的模型文件，如果存在则调用 load_model 函数加载模型。
#   - 将加载好的模型移动到指定设备（GPU 或 CPU），并用于绘制预测结果和模型总结。
#   - 通过用户选择决定是否加载已保存的模型，便于继续训练或进行预测任务。

# 这些详细注释有助于理解 load_model 函数的功能、具体步骤和在项目中的使用情况，便于后续的维护和使用。
