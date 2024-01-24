from typing import Any, Optional, Union, Tuple
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_hparams(path):
    hparam_type = {}
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    tags = event_acc.Tags()["tensors"]
    for tag in tags:
        name = tag.split('/')[0]
        event_list = event_acc.Tensors(tag)
        param_str = str(event_list[0].tensor_proto.string_val[0])
        param_str = param_str.replace('\\n', '')
        param_str = param_str.replace('\\t', '')
        param_str = param_str.replace('\'', '')
        param_str = param_str.replace('\\', '')
        param_str = param_str.replace('b{', '{')
        if param_str.startswith('{'):
            params = json.loads(param_str)
            hparam_type[name] = params
    if 'Param/LR' in event_acc.Tags()['scalars']:
        lr_events = event_acc.Scalars('Param/LR')
        last_lr = lr_events[-1].value
    else:
        last_lr = None
    if 'Param/Lengthscale' in event_acc.Tags()['scalars']:
        ls_events = event_acc.Scalars('Param/Lengthscale')
        last_ls = ls_events[-1].value
    elif 'Param/kernel_sigma' in event_acc.Tags()['scalars']:
        ls_events = event_acc.Scalars('Param/kernel_sigma')
        last_ls = ls_events[-1].value
    else:
        last_ls = None
    return hparam_type, last_lr, last_ls

def lead_lag_transform(data: torch.Tensor, t:torch.Tensor, lead_lag: int|list[int]=1):
    '''
    Transform data to lead-lag format
    data is of shape (seq_len, seq_dim)
    '''
    assert len(data.shape) == 2, 'data must be of shape (seq_len, seq_dim)'
    assert data.shape[0] == t.shape[0], 'data and df_index must have the same length'
    if isinstance(lead_lag, int):
        if lead_lag <= 0: raise ValueError('lead_lag must be a positive integer')
    else:
        for lag in lead_lag:
            if lag <= 0: raise ValueError('lead_lag must be a positive integer')

    # get shape of output
    seq_len = data.shape[0]
    seq_dim = data.shape[1]
    shape = list(data.shape)
    if isinstance(lead_lag, int):
        lead_lag = [lead_lag]
    max_lag = max(lead_lag)
    shape[0] = shape[0] + max_lag
    shape[1] = (len(lead_lag) + 1) * seq_dim

    # create time dimension
    t = torch.cat([t, torch.ones(max_lag) * t[-1]], dim=0).reshape(-1, 1) # pad latter values with last value, shape (seq_len + max_lag, 1)

    # NOTE: removed as transformation applied at get_item level of DFDataset
    # t = np.zeros(seq_len, dtype=data.dtype)
    # t[1:] = (df_index.to_series().diff()[1:].dt.days / 365).values.cumsum()
    # t = np.concatenate([t, np.ones(max_lag) * t[-1]]).reshape(-1, 1) # pad latter values with last value, shape (seq_len + max_lag, 1)

    # create lead-lag series
    lead_lag_data = torch.empty(shape, dtype=data.dtype) # shape (seq_len + max_lag, seq_dim * (len(lead_lag) + 1))
    lead_lag_data[:seq_len, :seq_dim] = data # fill in original sequence
    lead_lag_data[seq_len:, :seq_dim] = data[-1] # pad latter values with last value
    for i, lag in enumerate(lead_lag):
        i = i + 1 # skip first seq_dim columns
        lead_lag_data[:lag, i*seq_dim:(i+1)*seq_dim] = 0.0 # pad initial values with zeros
        lead_lag_data[lag:lag+seq_len, i*seq_dim:(i+1)*seq_dim] = data
        lead_lag_data[lag+seq_len-1:, i*seq_dim:(i+1)*seq_dim] = data[-1] # pad latter values with last value
    return torch.cat([t, lead_lag_data], axis=1)

def batch_lead_lag_transform(data: torch.Tensor, t:torch.Tensor, lead_lag: int|list[int]=1):
    '''
    Transform data to lead-lag format
    data is of shape (seq_len, seq_dim)
    '''
    assert data.ndim == 3 and t.ndim == 3, 'data and t must be of shape (batch_size, seq_len, seq_dim)'
    assert data.shape[1] == t.shape[1], 'data and df_index must have the same length'
    if isinstance(lead_lag, int):
        if lead_lag <= 0: raise ValueError('lead_lag must be a positive integer')
    else:
        for lag in lead_lag:
            if lag <= 0: raise ValueError('lead_lag must be a positive integer')

    # get shape of output
    batch_size = data.shape[0]
    seq_len = data.shape[1]
    seq_dim = data.shape[2]
    shape = list(data.shape)
    if isinstance(lead_lag, int):
        lead_lag = [lead_lag]
    max_lag = max(lead_lag)
    shape[1] = shape[1] + max_lag
    shape[2] = (len(lead_lag) + 1) * seq_dim

    # create time dimension t.shape = (batch_size, seq_len, 1)
    # pad latter values with last value, shape (seq_len + max_lag, 1)
    t = torch.cat([t, (torch.ones(batch_size, max_lag, 1, dtype=t.dtype, device=t.device, requires_grad=False) * t[:,-1:,:])], dim=1)

    # create lead-lag series
    lead_lag_data = torch.empty(shape, dtype=data.dtype, device=t.device, requires_grad=False) # shape (seq_len + max_lag, seq_dim * (len(lead_lag) + 1))
    lead_lag_data[:, :seq_len, :seq_dim] = data # fill in original sequence
    lead_lag_data[:, seq_len:, :seq_dim] = data[:,-1:,:] # pad latter values with last value
    for i, lag in enumerate(lead_lag):
        i = i + 1 # skip first seq_dim columns
        lead_lag_data[:, :lag, i*seq_dim:(i+1)*seq_dim] = 0.0 # pad initial values with zeros
        lead_lag_data[:, lag:lag+seq_len, i*seq_dim:(i+1)*seq_dim] = data
        lead_lag_data[:, lag+seq_len-1:, i*seq_dim:(i+1)*seq_dim] = data[:,-1:,:] # pad latter values with last value
    return torch.cat([t, lead_lag_data], axis=2)

class DFDataset(Dataset):
    '''
    Dataset for dataframes with time series
    Each sample is of shape (sample_len, seq_dim+1) where the first column is the time dimension if time_dim=True
    '''
    def __init__(self, df: pd.DataFrame, sample_len: int, scale:float, stride: int=1,
                 col_idx: Optional[int|list[int]]=None, dtype=torch.float32):

        self.dataset = torch.tensor(df.iloc[:,col_idx].values, dtype=dtype, requires_grad=False) # (seq_len, seq_dim)
        self.dataset[:,2:] = self.dataset[:,2:] / scale # scale from 0. to 12.7 for pitch and velocity
        self.sample_len = sample_len
        self.shape = self.dataset.shape
        self.stride = stride
        self.len = int((self.dataset.shape[0] - self.sample_len)/self.stride) + 1
        self.seq_dim = len(col_idx) if isinstance(col_idx, list) else 1
        self.scale = scale

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start = idx*self.stride
        end = start + self.sample_len
        path = torch.empty((self.sample_len, self.seq_dim), dtype=self.dataset.dtype, requires_grad=False) # shape (sample_len, seq_dim)
        path[:,:2] = self.dataset[start:end, :2] - self.dataset[start, 0] # set start time to 0
        path[:,2:] = self.dataset[start:end, 2:] # shape (sample_len, seq_dim)
        return path
