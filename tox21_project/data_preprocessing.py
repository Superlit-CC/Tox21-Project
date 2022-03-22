###########
# 作者：曹成
#
# 数据预处理部分
###########



import numpy as np
from collections import Counter
from tqdm import tqdm
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class CreateSymbolDataset(object):
    """分割smiles，并保存分割后的符号(symbol)和hash值(index)"""
    def __init__(self, smiles) -> None:
        self.smiles = smiles

    def _get_list_symbols(self) -> None:
        """按照字母来分割smiles"""
        symbols = set()
        for smile in self.smiles:
            # union 合并set
            symbols = symbols.union(set(list(smile)))
        
        symbols = list(symbols)
        self.symbols = symbols
        self.sym_idx = {symbols[i]:i for i in range(0, len(symbols))}

    def get_symbols(self):
        """返回分割后的symbol"""
        return self.symbols

    def get_sym_idx(self):
        """返回字典{symbol:index}"""
        return self.sym_idx

class BagOfWords(CreateSymbolDataset):
    def __init__(self, smiles) -> None:
        super(BagOfWords, self).__init__(smiles)

    def fit(self) -> np.array:
        """它计算每个smiles中的符号数量，并为每个smiles的symbols创建一个 np.array 计数"""
        self._get_list_symbols()
        # 下面的smiles有可能有训练集里没有的标识符，用 '_unk_' 表示
        self.symbols.append('_unk_')
        self.sym_idx['_unk_'] = len(self.symbols) - 1

        count = np.zeros((len(self.smiles), len(self.symbols)))
        for i, smile in enumerate(self.smiles):
            c = Counter(smile) # Counter直接将计数后的结果以字典返回
            for k, v in c.items():
                count[i][self.sym_idx[k]] += v
        return count

    def transform(self, smiles) -> np.array:
        """将输入的smiles按照fit函数来转换格式，比如转换测试集"""
        count = np.zeros((len(smiles), len(self.symbols)))
        for i, smile in enumerate(smiles):
            c = Counter(smile)
            for k, v in c.items():
                if k in self.symbols:
                    count[i][self.sym_idx[k]] += v
                else:
                    count[i][self.sym_idx['_unk_']] += v
        return count

class MorganFingerprints:
    def __init__(self, smiles) -> None:
        self.smiles = smiles
    
    def _get_fingerprint_from_smile(self, smile:str, radius:int) -> np.array:
        """从smile获取bit分子指纹"""
        mol = Chem.MolFromSmiles(smile)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius) # bit长度默认2048
        self.fingerprints.append(fingerprint)
        return np.array(list(fingerprint))

    def transform(self, radius:int=2) -> np.array:
        """将所有的smiles全部转化成指纹 并输出np.array"""
        res = []
        self.fingerprints = []
        for smile in self.smiles:
            fingerprint = self._get_fingerprint_from_smile(smile, radius)
            res.append(fingerprint)
        return np.array(res)

class MurckoScaffold:
    """分子骨架 返回类型为list[str]"""
    def __init__(self, smiles) -> None:
        self.smiles = smiles
    
    def get_scaffold(self, smile:str) -> str:
        """返回smile的骨架，如果没有，则返回原smile"""
        mc = MS.MurckoScaffoldSmilesFromSmiles(smile)
        return mc if mc else smile

    def transform(self) -> list:
        scaffold_smiles = []
        for smile in self.smiles:
            scaffold_smiles.append(self.get_scaffold(smile))
        return scaffold_smiles

class MolecularDescriptors:
    def __init__(self, smiles) -> None:
        self.smiles = smiles
        self.descriptors = Descriptors._descList # 类型为 list[(tuple)]
        
    def get_descriptors(self) -> np.array:
        """计算mol的所有分子描述符"""
        res = np.zeros((len(self.smiles), len(self.descriptors)))
        for i, smile in enumerate(tqdm(self.smiles)):
            mol = Chem.MolFromSmiles(smile)
            for j, (name, fun) in enumerate(self.descriptors):
                try:
                    value = fun(mol)
                except:
                    value = np.nan
                res[i][j] = value
        return res
            
