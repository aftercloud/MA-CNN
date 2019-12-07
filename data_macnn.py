import cv2
import h5py
import torch
import torch.optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 数据集参数
test_size = 106/606
batch_size = 10
random_state = 0
shuffle = True


# 数据集类
class MacnnDataset(Dataset):
    def __init__(self, x, y):
        super(MacnnDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        img = torch.tensor(cv2.resize(self.x[index], (448, 448))).float().unsqueeze(0)
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.x)


# 原始数据读取
f = h5py.File('bu3d_features.h5')
x, y = [], []
for index, name in enumerate(f):
    for file in f[name]:
        x.append(f[name][file].value)
        y.append(index)
f.close()


# 切割训练集和测试集
x, tx, y, ty = train_test_split(x, y, test_size=test_size, random_state=random_state)

# 构建数据加载器
train_set = MacnnDataset(x, y)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
test_set = MacnnDataset(tx, ty)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)
