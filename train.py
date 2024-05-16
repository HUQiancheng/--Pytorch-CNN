from src import *  # 从 src 包中导入所有模块
import os  # 引入 os 库用于文件和目录操作
import torch  # 引入 PyTorch 库
from torch.utils.data import DataLoader  # 引入 DataLoader 用于数据加载

def main():
    """
    主函数，负责训练和评估模型
    """
    root_dir = 'data/dataset'  # 数据集的根目录
    model_path = 'src/runs/cnn_model.pth'  # 保存模型的路径
    
    # 创建训练数据加载器和验证数据加载器
    # 调用 create_dataloader 函数，传入根目录、数据转换操作、batch_size 和是否为训练集的标志
    # create_dataloader 定义在 src/training/train_CNN.py 中
    trainloader = create_dataloader(root_dir, transform, batch_size=64, train=True)
    valloader = create_dataloader(root_dir, transform, batch_size=64, train=False)

    # 确定使用的设备（GPU 或 CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查是否存在已保存的模型
    if os.path.exists(model_path):
        # 如果找到已保存的模型，询问用户是否使用它
        use_saved_model = input("Saved model found. Do you want to use it? (y/n): ").strip().lower()
        if use_saved_model == 'y':
            model = load_model(model_path)  # 调用 load_model 函数加载已保存的模型
            model.to(device)  # 将模型移动到指定设备（GPU 或 CPU）
            print("Loaded saved model.")
            plot_predictions(model, valloader, device)  # 调用 plot_predictions 函数绘制模型预测结果
            summarize_model(model)  # 调用 summarize_model 函数总结并打印模型信息
            return  # 结束函数，跳过训练阶段
        else:
            print("Training a new model.")  # 训练新模型

    # 创建新的模型并移动到指定设备
    model = CNNModel().to(device)  # 实例化 CNNModel 并将其移动到指定设备
    summarize_model(model)  # 调用 summarize_model 函数总结并打印模型信息

    criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数，使用交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 定义优化器，使用随机梯度下降（SGD），学习率为 0.001，动量为 0.9

    # 训练模型
    # 调用 train 函数，传入模型、训练数据加载器、验证数据加载器、损失函数、优化器、设备和训练轮数
    # train 函数定义在 src/training/train_CNN.py 中
    model, history = train(model, trainloader, valloader, criterion, optimizer, device, epochs=5)

    # 保存训练好的模型
    torch.save(model.state_dict(), model_path)  # 将模型的状态字典保存到指定路径
    print(f"Model saved to {model_path}")

    # 绘制训练历史
    # 调用 plot_training_history 函数绘制训练和验证过程中的损失和准确率曲线
    plot_training_history(history)

    # 绘制模型预测结果
    # 调用 plot_predictions 函数，使用验证数据加载器绘制模型的预测结果
    plot_predictions(model, valloader, device)

if __name__ == '__main__':
    main()  # 调用主函数
