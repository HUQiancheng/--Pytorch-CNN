import os  # 用于处理文件和目录路径
from PIL import Image, ImageOps  # PIL(Pillow)库用于图像处理，Image用于打开和处理图像，ImageOps用于图像操作如颜色反转
import torch  # 引入PyTorch库，主要用于深度学习
from torch.utils.data import Dataset  # 从PyTorch的utils.data模块引入Dataset类，用于创建自定义数据集
from torchvision import transforms  # 从torchvision模块引入transforms，用于对图像数据进行转换和预处理
from sklearn.model_selection import train_test_split  # 从sklearn库中引入train_test_split函数，用于将数据集划分为训练集和验证集

# 定义自定义数据集类，继承自PyTorch的Dataset类
class CNN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, train_ratio=0.8):
        """
        初始化数据集

        参数:
        root_dir (str): 数据集的根目录
        transform (callable, optional): 应用于图像的转换函数
        train (bool): 指定是训练集还是验证集
        train_ratio (float): 训练集比例, 默认为0.8
        """
        self.root_dir = root_dir  # 保存数据集的根目录
        self.transform = transform  # 保存图像转换函数
        self.train = train  # 指定数据集是否用于训练
        self.image_files = []  # 初始化存储所有图像文件路径的列表
        self.labels = []  # 初始化存储所有图像对应标签的列表
        self._load_data(root_dir)  # 调用加载数据函数，加载图像文件路径和标签

        # 使用train_test_split函数将数据集划分为训练集和验证集
        train_files, val_files, train_labels, val_labels = train_test_split(
            self.image_files, self.labels, train_size=train_ratio, stratify=self.labels, random_state=42
        )
        if self.train:
            # 如果是训练集，使用划分后的训练数据
            self.image_files, self.labels = train_files, train_labels
        else:
            # 如果是验证集，使用划分后的验证数据
            self.image_files, self.labels = val_files, val_labels

    def _load_data(self, root_dir):
        """
        加载数据，将图像文件路径和标签存储到相应的列表中

        参数:
        root_dir (str): 数据集的根目录
        """
        for label in range(10):  # 遍历0到9的标签目录
            folder = os.path.join(root_dir, str(label))  # 获取每个标签目录的路径
            for root, _, files in os.walk(folder):  # 遍历标签目录中的文件
                for filename in files:  # 遍历每个文件
                    if filename.endswith('.png'):  # 仅处理以.png结尾的文件
                        self.image_files.append(os.path.join(root, filename))  # 将图像文件路径添加到列表中
                        self.labels.append(label)  # 将对应的标签添加到标签列表中

    def __len__(self):
        """
        返回数据集的大小

        返回:
        int: 数据集中图像的数量
        """
        return len(self.image_files)  # 返回图像文件数量

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签

        参数:
        idx (int): 图像和标签的索引

        返回:
        tuple: 包含图像和标签的元组
        """
        img_name = self.image_files[idx]  # 根据索引获取图像文件路径
        image = Image.open(img_name).convert("RGBA")  # 使用PIL打开图像并转换为RGBA模式

        # 创建一个白色背景的图像
        white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # 创建一个与原图像大小相同的白色背景
        white_background.paste(image, (0, 0), image)  # 将原始图像粘贴到白色背景上，保持原图像的透明度

        # 将图像转换为RGB模式以移除alpha通道
        image = white_background.convert("RGB")  # 将图像从RGBA模式转换为RGB模式

        # 反转图像颜色
        image = ImageOps.invert(image.convert('L'))  # 将图像转换为灰度图像，然后进行颜色反转

        label = self.labels[idx]  # 获取对应的标签
        if self.transform:
            # 如果有转换函数，应用于图像
            image = self.transform(image)
        return image, label  # 返回图像和标签

# 数据预处理和转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据，使其均值为0.5，标准差为0.5
])
