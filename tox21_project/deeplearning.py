#####################
# 作者：曹成
#
# 深度学习模型功能实现
#####################


from random import shuffle

from tqdm import tqdm, tqdm_notebook
import numpy as np
import pandas as pd
from collections import deque # collections 包含了一些特殊的容器，deque 双向队列

from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def get_data(train_X, train_y, train_mask, test_X, test_y, test_mask, batch_size):
    """将数据打包成可训练的数据集"""
    # TensorDataset 类似于 python 中的 zip 功能
    train_set = TensorDataset(torch.tensor(train_X), torch.tensor(train_y.values), torch.tensor(train_mask.values))
    test_set = TensorDataset(torch.tensor(test_X), torch.tensor(test_y.values), torch.tensor(test_mask.values))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    return train_set, test_set, train_loader

class net(nn.Module):
    """模型的构建，里面包括了残差层、随机种子等"""
    def __init__(self, input_size:int, output_size:int, hidden_layers:list=[64, 64], \
                    activation=torch.relu, drop_p:float=0.5, res_layer:bool=True, seed:int=12345) -> None:
        super(net, self).__init__()

        # configuration
        # manual_seed 保证这个py文件的输出结果相同
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        self.dropout = nn.Dropout(p=drop_p)
        self.activation = activation
        self.res_layer = res_layer

        # 隐藏层的实现
        # ModuleList 继承于nn.Module，可以作为Module子模块，会将模型参数保存用于训练，而list则不行
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])]) # 第一层
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:]) # n - 1 个线性的输入输出层（隐藏层）
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes]) # extend合并两个list
        self.constant_layers = nn.ModuleList([nn.Linear(h, h) for h in hidden_layers]) # 固定层

        # 输出层
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # batch norm层
        # BatchNorm1d对batch维度进行类归一化
        self.bat = nn.ModuleList(nn.BatchNorm1d(i) for i in hidden_layers)

    def forward(self, input):
        # 在每个隐藏层后面插入如下：
        for i, linear in enumerate(self.hidden_layers):
            # 1. 激活层
            input = self.activation(linear(input))
            # 2. 残差层：固定层 + input ，不加固定层就层数比较少，没必要有残差层
            if self.res_layer:
                input = self.activation(self.constant_layers[i](input)) + input
            else:
                input = self.activation(self.constant_layers[i](input))
            # 3. Batch Norm层
            input = self.bat[i](input)
            # 4. Drop out层
            input = self.dropout(input)
        
        # 输出层
        output = self.output(input)
        output = torch.sigmoid(output) # 二分类sigmoid
        return output

class EarlyStopping:
    """实现训练的提前停止功能"""
    def __init__(self, patience:int=10) -> None:
        self.patience = patience
        self.values = deque(maxlen=patience)
        self.current_max = -np.inf
    def to_stop(self, value) -> bool:
        # 1. 更新最大值
        self.current_max = max(self.current_max, value)
        # 2. 添加当前的value
        self.values.append(value)
        # 3. 如果当前最大值比双向队列里的所有值都大，即score不再增加反而减小，则停止
        if len(self.values) == self.patience and (np.array(self.values) < self.current_max).all():
            return True
        return False

class Trainer:
    """训练器"""
    def __init__(self, model, optimizer, criterion, epochs, device, scheduler=None, early_stop=None) -> None:
        self.model = model
        self.optimizer = optimizer # 优化器
        self.criterion = criterion # 目标函数
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler # 依据是否loss升高或降低来动态更新学习率，使用时，先声明类，再scheduler.step(test_acc)，括号中就是指标一般用验证集的loss
        self.early_stop = early_stop

    def _average_scores(self, y_true, y_prob, weight, score) -> float:
        """传入评价函数score，返回所有scores的平均值"""
        res = np.array([score(y_true[:, i], y_prob[:, i], sample_weight=weight[:, i]) for i in range(12)])
        return res.mean()

    def _training_routine(self, X, y, weight):
        """一个batch的训练过程：输入X, y, weight，返回模型的loss和output"""
        # 1. 模型梯度置0
        self.model.zero_grad()
        # 2. 计算预测值output
        output = self.model(X).to(self.device)
        # 3. 目标函数权重赋值
        self.criterion.weight = weight
        # 4. 计算loss
        loss = self.criterion(output, y).to(self.device)
        # 5. 根据loss反向传播梯度
        loss.backward()
        # 6. 参数更新
        self.optimizer.step()
        return loss, output

    def _creat_train_df(self):
        """建立df来存储每个epoch的训练数据"""
        df = pd.DataFrame(columns=['train loss', 'val loss', 'val auprc', 'val aucroc'], index=[i + 1 for i in range(self.epochs)])
        df.index = df.index.set_names('Epochs')
        return df

    def train_model(self, train_loader, valid_set, model_name, print_interval:int=10):
        minimum_val_loss = float('inf')
        df = self._creat_train_df()

        for epoch in tqdm_notebook(range(self.epochs)):
            epoch_loss = 0
            self.model.train()

            # 1. 从训练集里导入每个batch
            for X, y, weight in train_loader:
                # 2. 将数据导入到device中
                X = X.to(self.device).float()
                y = y.to(self.device).float()
                weight = weight.to(self.device).float()
                # 3. 应用_training_routine
                tloss, output = self._training_routine(X, y, weight)
                epoch_loss += tloss
            
            # 4. 在valid_set上评估模型
            self.model.eval()
            X_val, y_val, weight_val = valid_set[:]
            pred = self.model(X_val.to(self.device).float())
            self.criterion.weight = weight_val.to(self.device).float()
            vloss = self.criterion(y_val.to(self.device).float(), pred)

            # 5. 应用scheduler
            if self.scheduler is not None:
                self.scheduler.step(vloss)
            
            # 6. 计算验证集的AUPRC和AUCROC
            np_pred = pred.clone().detach().cpu().numpy()
            auprc = self._average_scores(y_val.numpy(), np_pred, weight_val.numpy(), average_precision_score)
            aucroc = self._average_scores(y_val.numpy(), np_pred, weight_val.numpy(), roc_auc_score)

            # 7. 打印保存结果
            if epoch % print_interval == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}]----train loss: {epoch_loss:.6f}  val loss: {vloss:.6f}  AUPRC: {auprc:.3f}  AUCROC: {aucroc:.3f}')
            # 保存模型的参数和优化器的参数
            torch.save({
                'epoch': epoch,
                'model': self.model,
                'optimizer': self.optimizer
                }, 'models/dnn_models/' + model_name + '.pth')
            
            # 8. early stop
            if self.early_stop is not None:
                if self.early_stop.to_stop(auprc):
                    print(f'由于AUPRC在{self.early_stop.patience}次epoch后没有增长')
                    return df.dropna()
            # '\r'将光标移动到行开头，end 默认结尾为'\n'
            # print('\r', f'Epoch [{epoch + 1}/{self.epochs}]', end='')
            df.loc[epoch + 1] = [epoch_loss.cpu().detach().numpy(), vloss.cpu().detach().numpy(), auprc, aucroc]
        return df
