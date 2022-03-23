############
# 作者：曹成
#
# 用于测试模型结果，并画图
############


# Imports
import numpy as np 
import pandas as pd 
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm_notebook, tqdm
import matplotlib.pyplot as plt

class panel_of_test:
    """两个功能：计算指标 画ROC图形"""
    def __init__(self, assays, y:np.array, sample_weights:np.array=None, notebook:bool=True) -> None:
        self.assays = assays
        self.y = y
        self.notebook = notebook # 是否在notebook中运行
        self.sample_weights = sample_weights
    
    def compute_basic_metrics(self, y_pred:np.array, y_scores:np.array) -> pd.core.frame.DataFrame:
        """计算所有任务的基础指标"""

        res = pd.DataFrame(index=self.assays, columns=['Precision','Recall', 'F1', 'AUPRC', 'Accuracy', 'Balanced Accuracy','ROC_AUC'])

        # tqdm_notebook专为notebook设置的进度条，leave进度条是否保留，默认为True
        if self.notebook:
            t = tqdm_notebook(range(len(self.assays)), leave=True)
        else:
            t = tqdm(range(len(self.assays)), leave=True)
        
        for i in t:
            precision = precision_score(self.y[:, i], y_pred[:, i], sample_weight=self.sample_weights[:, i])
            recall = recall_score(self.y[:, i], y_pred[:, i], sample_weight=self.sample_weights[:,i])
            f1 = f1_score(self.y[:, i], y_pred[:, i], sample_weight=self.sample_weights[:, i])
            auprc = average_precision_score(self.y[:, i], y_scores[:, i], sample_weight=self.sample_weights[:, i])
            acc = accuracy_score(self.y[:, i], y_pred[:, i], sample_weight=self.sample_weights[:, i])
            bal_acc = balanced_accuracy_score(self.y[:, i], y_pred[:, i], sample_weight=self.sample_weights[:, i])
            aucscore = roc_auc_score(self.y[:, i], y_scores[:, i], sample_weight=self.sample_weights[:, i])

            res.loc[self.assays[i]] = [precision, recall, f1, auprc ,acc, bal_acc, aucscore]
        return res

    def _plot_precision_recall_curve(self, y_test, y_scores, axs, axs_index_tuple, assays, assays_index):
        """给测试集和预测数据，同时给图和任务，以及它们的index，画出"""
        
        # 根据不同的threshold计算出不同的precision和recall
        precision, recall, thresholds =  precision_recall_curve(y_test, y_scores)

        # 子图的index
        i, j = axs_index_tuple
        axs[i][j].plot(recall, precision)
        axs[i][j].set_xlabel('Recall')
        axs[i][j].set_ylabel('Precision')
        axs[i][j].set_title(assays[assays_index])

    def _plot_roc_curve(self, y_test, y_scores, axs, axs_index_tuple, assays, assays_index):
        """给测试集和预测数据，同时给图和任务，以及它们的index，画出"""
        
        # 根据不同的threshold计算出不同的precision和recall
        fpr, tpr, thresholds =  roc_curve(y_test, y_scores, pos_label=1)

        # 子图的index
        i, j = axs_index_tuple
        # lw：折线图的线条宽度
        axs[i][j].plot(fpr, tpr, color='darkorange', lw=2)
        axs[i][j].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        axs[i][j].set_xlabel('FPR')
        axs[i][j].set_ylabel('TPR')
        axs[i][j].set_title(assays[assays_index])
    
    def plot_precision_recall(self, y_scores, extra_index:bool=False) -> None:
        """根据给的y和y_scores来画出所有任务的ROC曲线"""

        fig, axs = plt.subplots(4, 3, figsize=(20, 20))

        for i in range(len(self.assays)):
            plot_index_x = i // 3
            plot_index_y = i % 3
            axs_index_tuple = (plot_index_x, plot_index_y)
            # 是否有额外的维数
            if extra_index:
                yy_scores = y_scores[i][:, 1]
            else:
                yy_scores = y_scores[:, i]
            self._plot_precision_recall_curve(self.y[:, i], yy_scores, axs, axs_index_tuple, self.assays, i)
    
    def plot_roc(self, y_scores, extra_index:bool=False) -> None:
        """根据给的y和y_scores来画出所有任务的ROC曲线"""

        fig, axs = plt.subplots(4, 3, figsize=(20, 20))

        for i in range(len(self.assays)):
            plot_index_x = i // 3
            plot_index_y = i % 3
            axs_index_tuple = (plot_index_x, plot_index_y)
            # 是否有额外的维数
            if extra_index:
                yy_scores = y_scores[i][:, 1]
            else:
                yy_scores = y_scores[:, i]
            self._plot_roc_curve(self.y[:, i], yy_scores, axs, axs_index_tuple, self.assays, i)