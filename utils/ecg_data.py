import os
import pickle
import torch
import wfdb
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


drop_index = []

# Define ECG Graph Dataset
class ECGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) :
        return ["records100"]
    
    @property
    def processed_file_names(self) :
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        data_list = []

        edge_indexs = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 1, 5, 4, 4, 6, 6, 6, 6, 7, 7, 8, 10, 4, 10, 2, 3, 4, 5, 5, 5, 1, 10, 11, 3, 7, 8, 9, 8, 9, 9, 11, 11],
                                   [4, 11, 10, 2, 3, 4, 5, 5, 5, 1, 10, 11, 3, 7, 8, 9, 8, 9, 9, 11, 0, 0, 1, 1, 1, 1, 2, 1, 5, 4, 4, 6, 6, 6, 6, 7, 7, 8, 10, 0]])

        rawdata_path = "F:/chenteng/ASPP_AMGCN/data/ptb/raw/"
        X, label, _ = load_dataset(rawdata_path, 100)
        for idx in range(X.shape[0]):
            data = Data(x=torch.tensor(X[idx]).permute(1,0), edge_index=edge_indexs, y=label[idx])
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def select_dataset(dataset, Y):
    
    if len(drop_index) > 0:
        Y.drop(drop_index, inplace=True)
       
        print("===do drop===")
    
    Y.index = range(len(Y))

    train_dataset = dataset[list(Y[Y.strat_fold <= 8].index)]
    val_dataset = dataset[list(Y[Y.strat_fold == 9].index)]
    test_dataset = dataset[list(Y[Y.strat_fold == 10].index)]

    return train_dataset, val_dataset, test_dataset


def load_dataset(path, sampling_rate, release=False):

    # load and convert annotation data
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)
    agg_df = pd.read_csv(os.path.join(path, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    threshold = 0

    def agg(y_dic):
        temp = []
        for key, value in y_dic.items():
            if key in agg_df.index and value > threshold:
                c = agg_df.loc[key].diagnostic_class
                if str(c) != "nan":
                    temp.append(c)
        return list(set(temp))

    Y["diagnostic_superclass"] = Y.scp_codes.apply(agg)
    Y["superdiagnostic_len"] = Y["diagnostic_superclass"].apply(lambda x: len(x))
    counts = pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()
    Y["diagnostic_superclass"] = Y["diagnostic_superclass"].apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )

    X = X[Y["superdiagnostic_len"] >= 1]
    Y = Y[Y["superdiagnostic_len"] >= 1]

    mlb = MultiLabelBinarizer()
    mlb.fit(Y["diagnostic_superclass"])
    label = mlb.transform(Y["diagnostic_superclass"].values)

    return X, label, Y

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data





