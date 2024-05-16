import torch.nn as nn  # 引入 PyTorch 的神经网络模块

def summarize_model(model):
    """
    输出模型的详细信息，包括每一层的名称、参数的数量、参数的形状以及总参数数量

    参数:
    model (nn.Module): 需要总结的模型
    """
    print("Model Summary")
    print("=" * 100)
    # 打印表头，包括层的名称（Layer (type)）、输出形状（Output Shape）和参数数量（Param #）
    # 使用格式化字符串，使输出对齐
    # `print("{:<30} {:<30} {:<30}".format("Layer (type)", "Output Shape", "Param #"))`: 打印表头，包括 "Layer (type)"（层类型）、"Output Shape"（输出形状）和 "Param #"（参数数量），并使用 `format` 方法对齐列。
    print("{:<30} {:<30} {:<30}".format("Layer (type)", "Output Shape", "Param #"))
    print("=" * 100)

    total_params = 0  # 初始化总参数数量，用于累加模型的总参数数量

    # 遍历模型的所有参数，返回(name, parameter)元组
    for name, param in model.named_parameters():
        # 仅处理需要梯度计算的参数（可训练参数）
        if param.requires_grad:
            layer_param_count = param.numel()  # 获取参数的数量（元素个数）
            total_params += layer_param_count  # 累加到总参数数量
            # 打印每一层的详细信息，包括参数名称（name）、参数形状（param.shape）和参数数量（layer_param_count）
            print("{:<30} {:<30} {:<30}".format(name, str(param.shape), layer_param_count))

    print("=" * 100)
    print("Total parameters:", total_params)  # 打印总参数数量
    print("=" * 100)