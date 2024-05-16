import torch  # 引入 PyTorch 库，用于张量操作和模型管理
from PIL import Image  # 引入 PIL 库用于图像处理
from torchvision import transforms  # 引入 torchvision.transforms 用于图像转换和预处理
from src.model.CNNModel import CNNModel  # 从自定义模型文件中引入 CNNModel 类

def load_model(model_path):
    """
    加载训练好的模型

    参数:
    model_path (str): 模型路径

    返回:
    model: 加载的模型
    """
    model = CNNModel()  # 实例化自定义的 CNN 模型对象
    model.load_state_dict(torch.load(model_path))  # 加载模型的状态字典
    # `torch.load(model_path)` 会读取指定路径下的模型权重，并返回一个状态字典
    model.eval()  # 将模型设置为评估模式（关闭 dropout 和 batch normalization）
    return model  # 返回加载的模型

def predict(model, image_path):
    """
    对单张图片进行预测

    参数:
    model: 训练好的模型
    image_path (str): 图片路径

    返回:
    int: 预测结果（类别标签）
    """
    # 定义图像转换操作，用于预处理图像
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据，使其均值为0.5，标准差为0.5
    ])
    image = Image.open(image_path).convert('L')  # 打开图像并转换为灰度模式
    # `Image.open(image_path)` 打开图像文件，并使用 `convert('L')` 将其转换为灰度图像
    image = transform(image).unsqueeze(0)  # 将图像转换并增加一个批次维度
    # `transform(image)` 应用预处理操作，将图像转换为张量并标准化
    # `unsqueeze(0)` 增加一个维度，使其符合模型输入要求，即 [batch_size, channels, height, width]
    with torch.no_grad():  # 在不计算梯度的上下文中进行预测
        # `torch.no_grad()` 使预测过程不计算梯度，节省内存并加速计算
        output = model(image)  # 获取模型输出
        # 将预处理后的图像传入模型，得到输出张量
        _, predicted = torch.max(output, 1)  # 获取预测的类别
        # `torch.max(output, 1)` 返回每行最大值及其索引，`predicted` 是索引即类别标签
    return predicted.item()  # 返回预测的类别标签

if __name__ == '__main__':
    # 如果作为脚本执行，则运行以下代码

    # 加载已保存的模型
    model = load_model('src/runs/cnn_model.pth')  # 指定模型路径并加载模型
    
    # 设置待预测图像的路径
    image_path = 'path_to_test_image.png'  # 替换为实际的图像路径
    
    # 进行预测
    prediction = predict(model, image_path)  # 使用模型对图像进行预测
    
    # 打印预测结果
    print(f'Predicted label: {prediction}')  # 输出预测的类别标签
