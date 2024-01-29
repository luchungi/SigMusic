from typing import Any, Optional, Union, Tuple
import os
from os.path import basename, dirname, join, exists, splitext
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import xml.etree.ElementTree as ET
import xmldataset
import ipdb

from .midi import *

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

def batch_rectilinear_transform(data: torch.Tensor):
    '''
    Transform data to rectilinear format
    Data is of shape (batch_size, seq_len, 2) where the last dimension is (duration, note) format
    '''
    batch_size = data.shape[0]
    seq_len = data.shape[1]
    t = torch.zeros((batch_size, seq_len+1), dtype=data.dtype, device=data.device, requires_grad=False)
    t[:,1:] = torch.cumsum(data[:,:,0], dim=1) # shape (batch_size, seq_len)
    t = torch.cat([t[:,0].unsqueeze(1), torch.repeat_interleave(t[:,1:-1], 2, dim=1), t[:,-1].unsqueeze(1)], dim=1)
    tensor = torch.cat([t.unsqueeze(-1), torch.repeat_interleave(data[:,:,1:], 2, dim=1)], dim=-1)
    return tensor

def rectilinear_transform(df: pd.DataFrame, include_velocity: bool=False):
    rectilinear_path = []
    # gaps = df['Start'].iloc[1:].values - df['End'].iloc[:-1].values
    end_idx = 4 if include_velocity else 3
    rest = [0., 0.] if include_velocity else [0.]
    for i, row in df.iterrows():
        if i > 0:
            prev_end = rectilinear_path[-1][0]
            if prev_end < row['Start']:
                rectilinear_path.append([prev_end] + rest)
                rectilinear_path.append([row['Start']] + rest)
        rectilinear_path.append([row['Start']] + row[2:end_idx].values.tolist())
        rectilinear_path.append([row['End']] + row[2:end_idx].values.tolist())
    return np.array(rectilinear_path)

def note_duration_transform(dfs: list[pd.DataFrame]):
    '''
    Transform df to have 2 cols: duration and pitch
    0 is assumed to be reserved for rest with this transformation
    '''
    new_dfs = []
    for df in dfs:
        df = df.copy()
        rows = []
        df.loc[:, 'Pitch'] += 1 # add 1 to pitch to reserve 0 for rest
        prev_end = None
        for i, row in df.iterrows():
            note = pd.Series({'Duration': row['End'] - row['Start'], 'Pitch': row['Pitch']})
            if i == 0:
                rows.append(note)
                continue

            # check if previous note ends before current note starts i.e. gap between notes
            if prev_end is not None:
                if prev_end < row['Start']:
                    # add row with duration of gap with pitch 0 (rest)
                    rows.append(pd.Series({'Duration': row['Start'] - prev_end, 'Pitch': 0}))
                elif prev_end > row['Start']:
                    raise ValueError('Overlapping notes')
            prev_end = row['End']
            rows.append(note)
        new_dfs.append(pd.DataFrame(rows))
    return new_dfs

def get_dfs_from_midi(dir: str,
                      min_notes: int=1,
                      min_gap: float=0,
                      note_dur_transform: bool=False):
    '''
    Get dataframes from midi files in dir
    Filter out midi files where number of notes is less than min_notes
    Filter out midi files where minimum gap between notes is less than min_gap
    Transform dataframes to have 2 cols: duration and pitch if note_dur_transform is True
    0 is assumed to be reserved for rest with this transformation
    '''
    dfs = []
    for entry in os.scandir(dir):
        if entry.is_dir():
            dfs.extend(get_dfs_from_midi(entry.path, min_notes, min_gap, note_dur_transform))
        elif entry.is_file() and (entry.name.endswith('.midi') or entry.name.endswith('.mid')):
            midi_data = pretty_midi.PrettyMIDI(entry.path)
            df = midi_to_df(midi_data)

            # filter out midi files where gap between notes is less than min_gap
            gap = df['Start'].iloc[1:].values - df['End'].iloc[:-1].values
            if len(gap) ==0 or gap.min() < min_gap: continue

            if note_dur_transform:
                rows = []
                df.loc[:, 'Pitch'] += 1 # add 1 to pitch to reserve 0 for rest
                prev_end = None
                for i, row in df.iterrows():
                    note = pd.Series({'Duration': row['End'] - row['Start'], 'Pitch': row['Pitch']})
                    if i == 0:
                        rows.append(note)
                        continue

                    # check if previous note ends before current note starts i.e. gap between notes
                    if prev_end is not None:
                        if prev_end < row['Start']:
                            # add row with duration of gap with pitch 0 (rest)
                            rows.append(pd.Series({'Duration': row['Start'] - prev_end, 'Pitch': 0}))
                        elif prev_end > row['Start']:
                            raise ValueError('Overlapping notes')
                    prev_end = row['End']
                    rows.append(note)
                df = pd.DataFrame(rows)

            # filter out midi files where number of notes is less than min_notes
            if len(df) < min_notes: continue

            dfs.append(df) # only append if midi file contains notes
    return dfs

def trim_by_range(dfs: list[pd.DataFrame], min_range: int, max_range: int, exclude_rest: bool=False):
    '''
    Trim dataframes to have max_range pitch range
    '''
    new_dfs = []
    for df in dfs:
        range_of_pitch = df.loc[df['Pitch'] > 0, 'Pitch'].max() - df.loc[df['Pitch'] > 0, 'Pitch'].min() if exclude_rest else df['Pitch'].max() - df['Pitch'].min()
        if range_of_pitch >= min_range and range_of_pitch <= max_range:
            new_dfs.append(df.copy())
    return new_dfs

def move_octaves(dfs: list[pd.DataFrame],
                 min_pitch: Optional[int]=None,
                 max_pitch: Optional[int]=None,
                 center_range: Optional[list[int]]=None,
                 exclude_rest: bool=False):
    # assert either min_pitch and max_pitch are both not None or center is not None but not both
    assert (min_pitch is not None and max_pitch is not None) or center_range is not None, 'Either min_pitch and max_pitch are provided or center_range is provided'
    assert not((min_pitch is not None and max_pitch is not None) and center_range is not None), 'Either min_pitch and max_pitch are provided or center_range is provided'

    new_dfs = []
    for df in dfs:
        df_copy = df.copy()
        if center_range is not None:
            center = (df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'].max() + df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'].min()) / 2 if exclude_rest else (df_copy['Pitch'].max() + df_copy['Pitch'].min()) / 2
            while center < center_range[0]:
                df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'] += 12
                center += 12
            while center > center_range[1]:
                df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'] -= 12
                center -= 12
        if min_pitch is not None:
            curr_min = df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'].min() if exclude_rest else df_copy['Pitch'].min()
            while curr_min < min_pitch:
                df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'] += 12
                curr_min = df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'].min() if exclude_rest else df_copy['Pitch'].min()
        if max_pitch is not None:
            curr_max = df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'].max() if exclude_rest else df_copy['Pitch'].max()
            while curr_max > max_pitch:
                df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'] -= 12
                curr_max = df_copy.loc[df_copy['Pitch'] > 0, 'Pitch'].max() if exclude_rest else df_copy['Pitch'].max()
        new_dfs.append(df_copy)
    return new_dfs

def pitch_range(dfs: list[pd.DataFrame]):
    '''
    Print and plot pitch ranges
    '''
    pitch_ranges = []
    pitch_mins = []
    pitch_maxs = []
    for df in dfs:
        pitch_mins.append(df.loc[df['Pitch'] > 0, 'Pitch'].min()) # exclude 0 (rest)
        pitch_maxs.append(df.loc[df['Pitch'] > 0, 'Pitch'].max())
        pitch_range = df['Pitch'].max() - df.loc[df['Pitch'] > 0, 'Pitch'].min()
        pitch_ranges.append(pitch_range)
    pitch_mins = np.array(pitch_mins)
    pitch_maxs = np.array(pitch_maxs)
    pitch_ranges = np.array(pitch_ranges)
    min_pitch = pitch_mins.min()
    max_pitch = pitch_maxs.max()
    print('Min pitch:', min_pitch)
    print('Max pitch:', max_pitch)

    _, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].hist(pitch_ranges, bins=128, range=(1, 129));
    ax[0].set_title('Pitch range histogram')
    ax[1].boxplot(pitch_ranges);
    ax[1].set_title('Pitch range boxplot')
    ax[2].boxplot((pitch_maxs+pitch_mins) / 2);
    ax[2].set_title('Pitch range centers boxplot')
    plt.show()

    _, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(len(pitch_mins)), height=pitch_maxs-pitch_mins, bottom=pitch_mins, width=1.0);
    ax.scatter(range(len(pitch_mins)), (pitch_maxs+pitch_mins) / 2, color='red');
    plt.show()

    return min_pitch, max_pitch

def pitch_translation(dfs: list[pd.DataFrame]):
    '''
    Print and plot pitch ranges
    Translate pitch values to be between 1 and max pitch appearing in data
    0 is assumed to be reserved for rest
    '''
    min_pitch, max_pitch = pitch_range(dfs) # get min and max pitch appearing in data
    # translate pitch values to be between 1 and max pitch appearing in data
    for df in dfs:
        df.loc[df['Pitch'] > 0, 'Pitch'] -= (min_pitch - 1) # subtract min_pitch and add 1 to reserve 0 for rest
    return dfs, max_pitch - min_pitch + 1

class MIDIDataset(Dataset):
    '''
    Dataset for dataframes with MIDI data: 5 columns (start time, end time, pitch, velocity, instrument) in this order
    '''
    def __init__(self, dfs: list[pd.DataFrame], sample_len: int, cols: list[int]=[0,1,2,3], scale: float=1., stride: int=1, rectilinear: bool=False):

        self.seq_dim = len(cols) - 1 if rectilinear else len(cols) # start and end time are combined into one dimension for rectilinear
        self.cols = cols
        assert 0 in cols and 1 in cols, 'start time and end time column must be included'
        assert 2 in cols, 'pitch column must be included'
        self.sample_len = sample_len
        self.scale = scale
        self.stride = stride
        self.rectilinear = rectilinear

        self.tensors = []
        self.lens = []
        for df in dfs:
            if len(df) >= sample_len:
                if rectilinear:
                    rectilinear_path = rectilinear_transform(df, include_velocity=(3 in cols)) # shape (n_points, 2 or 3) depending on whether velocity is included
                    tensor = torch.tensor(rectilinear_path, dtype=torch.float32, requires_grad=False)
                    tensor[:,1:] = (tensor[:,1:] + 1.) / scale # add 1 to pitch and velocity which are integers from 0 to 127 to reserve 0 for rest and divide by scale
                else:
                    tensor = torch.tensor(df.iloc[:,:self.seq_dim].values, dtype=torch.float32, requires_grad=False)# (seq_len, seq_dim)
                    tensor[:,2:] = tensor[:,2:] / scale # scale pitch and velocity which are integers from 0 to 127
                self.tensors.append(tensor)
                self.lens.append(int((tensor.shape[0] - self.sample_len)/self.stride) + 1)
        self.lens = np.cumsum(self.lens)
        self.len = self.lens[-1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i = np.argmax(self.lens > idx) # get first argument in self.lens > idx
        idx = idx if i == 0 else idx - self.lens[i-1] # get index relative to start of tensor
        start = idx*self.stride
        end = start + self.sample_len
        path = torch.empty((self.sample_len, self.seq_dim), dtype=torch.float32, requires_grad=False) # shape (sample_len, seq_dim)
        if self.rectilinear:
            path[:,:1] = self.tensors[i][start:end,:1] - self.tensors[i][start, 0] # get start and end time columns and offset so that first start time is 0
            path[:,1:] = self.tensors[i][start:end,1:] # get pitch and velocity columns
        else:
            path[:,:2] = self.tensors[i][start:end,:2] - self.tensors[i][start, 0] # get start and end time columns and offset so that first start time is 0
            path[:,2:] = self.tensors[i][start:end, 2:] # get pitch and velocity columns
        return path

class NoteDurationDataset(Dataset):
    '''
    Dataset for dataframes with MIDI data: 4 columns (start time, end time, pitch) in this order
    '''
    def __init__(self, dfs: list[pd.DataFrame], sample_len: int, scale: float=1., stride: int=1):

        self.seq_dim = 2
        self.scale = scale
        self.stride = stride
        self.sample_len = sample_len
        self.tensors = []
        self.lens = []
        self.max_pitch = 1
        for df in dfs:
            if len(df) >= sample_len:
                self.max_pitch = max(self.max_pitch, df['Pitch'].max())
                tensor = torch.tensor(df.values, dtype=torch.float32, requires_grad=False)# (seq_len, seq_dim)
                tensor[:,1:] = tensor[:,1:] / scale # pitch is an integer starting from 1 and ending at max pitch
                self.tensors.append(tensor)
                self.lens.append(int((tensor.shape[0] - self.sample_len)/self.stride) + 1)
        self.lens = np.cumsum(self.lens)
        self.len = self.lens[-1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i = np.argmax(self.lens > idx) # get first argument in self.lens > idx
        idx = idx if i == 0 else idx - self.lens[i-1] # get index relative to start of tensor
        start = idx*self.stride
        end = start + self.sample_len
        path = self.tensors[i][start:end] # shape (sample_len, seq_dim)
        return path

def tensor_to_df(tensor: torch.Tensor, increment: int):
    tensor = tensor.cpu().detach().numpy()
    end = tensor[:,:,0].cumsum(axis=1)
    start = np.zeros_like(end)
    start[:,1:] = end[:,:-1]
    velocity = 80 * np.ones_like(tensor[:,:,0])
    pitch = tensor[:,:,1]
    pitch[pitch > 0] += increment

    dfs = []
    for i in range(tensor.shape[0]):
        df = pd.DataFrame({'Start': start[i], 'End': end[i], 'Pitch': pitch[i], 'Velocity': velocity[i]})
        df = df[df['Pitch'] > 0]
        dfs.append(df)

    return dfs

##############################################################################################################
# from https://github.com/annahung31/MidiNet-by-pytorch/blob/master/get_data.py

def get_sample(cur_song, cur_dur,n_ratio, dim_pitch, dim_bar):

    cur_bar =np.zeros((1,dim_pitch,dim_bar),dtype=int)
    idx = 1
    sd = 0
    ed = 0
    song_sample=[]

    while idx < len(cur_song):
        cur_pitch = cur_song[idx]-1
        ed = int(ed + cur_dur[idx]*n_ratio)
        # print('pitch: {}, sd:{}, ed:{}'.format(cur_pitch, sd, ed))
        if ed <dim_bar:
            cur_bar[0,cur_pitch,sd:ed]=1
            sd = ed
            idx = idx +1
        elif ed >= dim_bar:
            cur_bar[0,cur_pitch,sd:]=1
            song_sample.append(cur_bar)
            cur_bar =np.zeros((1,dim_pitch,dim_bar),dtype=int)
            sd = 0
            ed = 0
            # print(cur_bar)
            # print(song_sample)
        # if idx == len(cur_song)-1 and np.sum(cur_bar)!=0:
        #     song_sample.append(cur_bar)
    return song_sample

def build_matrix(note_list_all_c,dur_list_all_c):
    data_x = []
    prev_x = []
    zero_counter = 0
    for i in range(len(note_list_all_c)):
        song = note_list_all_c[i]
        dur = dur_list_all_c[i]
        song_sample = get_sample(song,dur,4,128,128)
        np_sample = np.asarray(song_sample)
        if len(np_sample) == 0:
            zero_counter +=1
        if len(np_sample) != 0:
            np_sample =np_sample[0]
            np_sample = np_sample.reshape(1,1,128,128)

            if np.sum(np_sample) != 0:
                place = np_sample.shape[3]
                new=[]
                for i in range(0,place,16):
                    new.append(np_sample[0][:,:,i:i+16])
                new = np.asarray(new)  # (2,1,128,128) will become (16,1,128,16)
                new_prev = np.zeros(new.shape,dtype=int)
                new_prev[1:, :, :, :] = new[0:new.shape[0]-1, :, :, :]
                data_x.append(new)
                prev_x.append(new_prev)

    data_x = np.vstack(data_x)
    prev_x = np.vstack(prev_x)


    return data_x,prev_x,zero_counter

def check_melody_range(note_list_all,dur_list_all):
    in_range=0
    note_list_all_c = []
    dur_list_all_c = []

    for i in range(len(note_list_all)):
        song = note_list_all[i]
        if len(song[1:]) ==0:
            ipdb.set_trace()
        elif min(song[1:])>= 60 and max(song[1:])<= 83:
            in_range +=1
            note_list_all_c.append(song)
            dur_list_all_c.append(dur_list_all[i])
    np.save('dur_list_all_c.npy',dur_list_all_c)
    np.save('note_list_all_c.npy',note_list_all_c)

    return in_range,note_list_all_c,dur_list_all_c

def transform_note(c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list):
    scale = [48,50,52,53,55,57,59,60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86,88,89,91,93]
    transfor_list_C1 = scale[0:7]
    transfor_list_C2 = scale[7:14]
    transfor_list_C3 = scale[14:21]

    transfor_list_D1 = scale[1:8]
    transfor_list_D2 = scale[8:15]
    transfor_list_D3 = scale[15:22]

    transfor_list_E1 = scale[2:9]
    transfor_list_E2 = scale[9:16]
    transfor_list_E3 = scale[16:23]

    transfor_list_F1 = scale[3:10]
    transfor_list_F2 = scale[10:17]
    transfor_list_F3 = scale[17:24]

    transfor_list_G1 = scale[4:11]
    transfor_list_G2 = scale[11:18]
    transfor_list_G3 = scale[18:25]

    transfor_list_A1 = scale[5:12]
    transfor_list_A2 = scale[12:19]
    transfor_list_A3 = scale[19:26]

    transfor_list_B1 = scale[6:13]
    transfor_list_B2 = scale[13:20]
    transfor_list_B3 = scale[20:27]

    note_c =[]
    dur_c =[]
    for file_ in c_key_list:
        note_list = [file_]
        dur_list = [file_]
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_C1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_C2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_C3[note-1]
                    note_list.append(h_idx)

                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_c.append(note_list)
            dur_c.append(dur_list)

        except:
            print('c key but no melody/notes :{}'.format(file_))

    note_d = []
    dur_d = []
    for file_ in d_key_list:
        note_list = [file_]
        dur_list = [file_]
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_D1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_D2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_D3[note-1]
                    note_list.append(h_idx)

                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_d.append(note_list)
            dur_d.append(dur_list)

        except:
            print('d key but no melody/notes :{}'.format(file_))

    note_e = []
    dur_e = []
    for file_ in e_key_list:
        note_list = [file_]
        dur_list = [file_]
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_E1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_E2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_E3[note-1]
                    note_list.append(h_idx)

                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_e.append(note_list)
            dur_e.append(dur_list)

        except:
            print('e key but no melody/notes :{}'.format(file_))

    note_f = []
    dur_f = []
    for file_ in e_key_list:
        note_list = [file_]
        dur_list = [file_]
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_F1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_F2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_F3[note-1]
                    note_list.append(h_idx)

                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_f.append(note_list)
            dur_f.append(dur_list)

        except:
            print('f key but no melody/notes :{}'.format(file_))


    note_g = []
    dur_g = []
    for file_ in a_key_list:
        note_list = [file_]
        dur_list = [file_]
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_G1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_G2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_G3[note-1]
                    note_list.append(h_idx)

                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_g.append(note_list)
            dur_g.append(dur_list)

        except:
            print('g key but no melody/notes :{}'.format(file_))

    note_a = []
    dur_a = []
    for file_ in a_key_list:
        note_list = [file_]
        dur_list = [file_]
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_A1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_A2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_A3[note-1]
                    note_list.append(h_idx)

                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_a.append(note_list)
            dur_a.append(dur_list)

        except:
            print('e key but no melody/notes :{}'.format(file_))


    note_b = []
    dur_b = []
    for file_ in a_key_list:
        note_list = [file_]
        dur_list = [file_]
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_B1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_B2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_B3[note-1]
                    note_list.append(h_idx)

                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_b.append(note_list)
            dur_b.append(dur_list)

        except:
            print('b key but no melody/notes :{}'.format(file_))


    note_list_all = note_c + note_d + note_e + note_f + note_g + note_a + note_b
    dur_list_all = dur_c + dur_d + dur_e  + dur_f + dur_g + dur_a  + dur_b

    return note_list_all,dur_list_all

def get_key(list_of_four_beat):
    key_list =[]
    c_key_list = []
    d_key_list = []
    e_key_list = []
    f_key_list = []
    g_key_list = []
    a_key_list = []
    b_key_list = []
    for file_ in list_of_four_beat:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            key = root.findall('.//key')
            key_list.append(key[0].text)
            if key[0].text == 'C':
                c_key_list.append(file_)
            if key[0].text == 'D':
                d_key_list.append(file_)
            if key[0].text == 'E':
                e_key_list.append(file_)
            if key[0].text == 'F':
                f_key_list.append(file_)
            if key[0].text == 'G':
                g_key_list.append(file_)
            if key[0].text == 'A':
                a_key_list.append(file_)
            if key[0].text == 'B':
                b_key_list.append(file_)
        except:
            print('file broken')
    # print('A key: {}'.format(key_list.count('A')))
    # print('B key: {}'.format(key_list.count('B')))
    # print('C key: {}'.format(key_list.count('C')))
    # print('D key: {}'.format(key_list.count('D')))
    # print('E key: {}'.format(key_list.count('E')))
    # print('F key: {}'.format(key_list.count('F')))
    # print('G key: {}'.format(key_list.count('G')))

    return c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list

def beats_(list_):
    list_of_four_beat =[]
    for file_ in list_:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            beats = root.findall('.//beats_in_measure')
            num = beats[0].text
            if num == '4':
                list_of_four_beat.append(file_)
        except:
            print('cannot open the file')
    return list_of_four_beat

def check_chord_type(list_file):
    list_ = []
    for file_ in list_file:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            check_list = []
            counter = 0
            None_counter = 0
            for item in root.iter(tag='fb'):
                check_list.append(item.text)
                counter +=1
                if item.text == None:
                    None_counter +=1
            for item in root.iter(tag='borrowed'):
                check_list.append(item.text)
                counter +=1
                if item.text == None:
                    None_counter +=1
            #print(check_list)
            #print(counter)
            #print(None_counter)
            if counter == None_counter :
                list_.append(file_)
        except:
            print('cannot open')
    return list_

def get_listfile(dataset_path):

    list_file=[]

    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if splitext(f)[0]=='chorus':
                fp = join(root, f)
                list_file.append(fp)

    return list_file
