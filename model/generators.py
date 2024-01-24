import torch
import torch.nn as nn
from torch.distributions import categorical, exponential

class TransformerMusic(nn.Module):
    def __init__(self, seq_dim:int, seq_len:int, hist_len:int, scale:float,
                 kernel_size:int|list[int], stride:int|list[int], n_channels:int,
                 n_head:int, n_transformer_layers:int, hidden_size:int,
                 activation:str):
        '''
        Transformer architecture with a deterministic output
        Uses a conv1d layer to encode the input sequence into an embedding
        Embedding is then fed into a transformer encoder
        Transformer encoder output is then fed into a linear layer to produce the output sequence (sequence of log returns)
        Noise is concatenated with output of the transformer encoder and then fed into a MLP to produce the final log returns
        Log returns are then cumsummed to produce the log path
        :param seq_dim: dimension of the time series e.g. how many stocks
        :param seq_len: length of the time series (technically can be changed to any length post training)
        :param hist_len: length of the historical path to feed to the network (length of returns = hist_len-1)
        :param kernel_size: kernel size of the conv1d layer
        :param stride: stride of the conv1d layer
        :param n_channels: number of channels of the conv1d layer
        :param n_head: number of heads of the transformer encoder
        :param n_transformer_layers: number of transformer layers
        :param activation: activation function to use
        '''
        super().__init__()
        self.seq_dim = seq_dim # dimension of the time series e.g. how many stocks
        self.seq_len = seq_len # length of the time series to generate i.e. does not include the historical data NOTE: different to LSTMRealdt
        self.hist_len = hist_len # length of historical path to feed to network (length of returns = hist_len-1)
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.stride = stride
        self.n_head = n_head
        self.n_transformer_layers = n_transformer_layers
        self.hidden_size = hidden_size
        self.scale = scale
        activation = getattr(nn, activation)

        self.encoded_seq_len = int((((hist_len) - kernel_size) / stride + 1))
        self.encoder = nn.Conv1d(seq_dim, n_channels, kernel_size=kernel_size, stride=stride) # +1 for dt dimension, n_channels is the embedding dim
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=n_channels, nhead=n_head, dim_feedforward=hidden_size,
                                                                    dropout=0., batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, n_transformer_layers)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoded_seq_len * n_channels, hidden_size*2),
            activation(),
            nn.Linear(hidden_size*2, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1+1+128+128), # (exp rate for start, exp rate for end, one hot categorical for pitch, one hot categorical for velocity)
        )

    def _generate_sequence(self, x): # (batch_size, sample_len, seq_dim))
        seq = []
        hist_x = x[:,:self.hist_len,:].permute(0,2,1) # (batch_size, seq_dim, hist_len)

        for _ in range(self.seq_len-self.hist_len):
            last_start = hist_x[:,0:1,-1] # (batch_size, 1) assumes start is the first dimension
            z = self.encoder(hist_x).permute(0,2,1) # permute back to (batch_size, encoded_seq_len, n_channels)
            z = self.transformer_encoder(z) # (batch_size, encoded_seq_len, n_channels)
            z = self.decoder(z) # (batch_size, 1+1+128+128)

            start_rate = z[:,0:1] # (batch_size, 1)
            end_rate = z[:,1:2]
            pitch = z[:,2:2+128] # (batch_size, 128)
            velocity = z[:,2+128:]

            start_dist = exponential.Exponential(torch.exp(start_rate))
            end_dist = exponential.Exponential(torch.exp(end_rate))
            pitch_dist = categorical.Categorical(logits=pitch)
            velocity_dist = categorical.Categorical(logits=velocity)

            start = start_dist.rsample() + last_start # (batch_size, 1)
            end = end_dist.rsample() + start # (batch_size, 1)
            pitch = pitch_dist.sample().unsqueeze(-1) / self.scale
            velocity = velocity_dist.sample().unsqueeze(-1) / self.scale

            note = torch.cat([start, end, pitch, velocity], dim=-1).unsqueeze(1) #(batch_size, 1, 4)
            seq.append(note)

            hist_x = hist_x.roll(-1, dims=2)
            hist_x[:,:,-1:] = note.permute(0,2,1)

        return torch.cat(seq, dim=1)

    def forward(self, x):
        return self._generate_sequence(x)