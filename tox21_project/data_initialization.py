#######
# 作者：曹成
#
# 导入数据，并执行基本的数据清理
#######


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
# 可以读取SDF格式
from rdkit.Chem import PandasTools
# 进行重复运算的函数
from functools import reduce

class dataInitialization:
    def __init__(self, filename:str):
        # LoadSDF参数有重命名，包括分子指纹，但似乎并没有分子指纹，麻了
        self.source = PandasTools.LoadSDF(filename, smilesName='SMILES', molColName='Molecule', includeFingerprints=True)
    
    def _clean_num(self, s:str)->float:
        """提取'(' 或 ' ' 或 整个数字 左边的第一个数字，返回float"""
        res = re.split(r'[\s(]', s)
        return float(res[0])

    def _add(self, x:float, y:float)->float:
        """
        两个数合并的规则如下 ：
        1. 有一个不为nan，则返回那个数
        2. 两个都为nan，则返回nan
        3. 有一个为1则返回1，否则返回0
        """
        if np.isnan(x) and np.isnan(y):
            return x
        elif not np.isnan(x) and np.isnan(y):
            return x
        elif np.isnan(x) and not np.isnan(y):
            return y
        elif x == y:
            return x
        else:
            return 1.0

    def _add_pd(self, series:pd.core.series.Series)->pd.core.series.Series:
        """传入一个series列，将重复元素合并，返回series类型"""
        return reduce(lambda x, y: self._add(x,y), series)

    def _find_duplicates(self, column_name:str)->np.array:
        """返回重复行的column_name"""
        return self.source[column_name][self.source[column_name].duplicated()].values
    
    def change_types(self, column_names:list, type)->None:
        """转换列的数据类型为type"""
        # astype 传入字典{k:v, ...}，将列名k的数据类型转换为v
        self.source = self.source.astype({t:type for t in column_names})

    def clean_num(self, column_names:list)->None:
        """column_names列都应用_clean_num函数"""
        for cn in column_names:
            self.source[cn] = self.source[cn].apply(lambda x: self._clean_num(x))
    
    def merge_duplicate_target_rows(self, duplicate_column:str, target_columns:list)->None:
        """合并重复元素"""
        # 1. 找到重复的duplicate_column
        duplicate_rows = self._find_duplicates(duplicate_column)
        # 2. 合并每一个重复的行
        for d in tqdm(duplicate_rows):
            # 2.1 将合并的行保存在temp中
            temp = self.source[self.source[duplicate_column] == d][target_columns].apply(self._add_pd) # apply默认在列上执行
            indx = list(self.source[self.source[duplicate_column] == d].index)
            keep_i = indx[0]
            drop_i = indx[1:]
            # 2.2 保留第一行
            temp2 = self.source.loc[keep_i]
            temp2.update(temp)
            self.source.loc[keep_i] = temp2
            # 下面这行代码无法更新行
            # self.source.loc[keep_i].update(temp)
            # 2.3 删除其余行
            for i in drop_i:
                self.source = self.source.drop(index=i)

    def save(self, filename_csv:str, filename_SDF:str=None):
        self.source.to_csv(filename_csv)
        if filename_SDF:
            PandasTools.WriteSDF(self.source, filename_SDF, molColName='Molecule', properties=list(self.source.columns))
