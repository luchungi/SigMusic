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

        return torch.cat(seq, dim=1) # (batch_size, seq_len, 4)

    def forward(self, x):
        return self._generate_sequence(x)

class LSTMusic(nn.Module):
    def __init__(self, seq_dim, seq_len, hidden_size=64, n_lstm_layers=1, activation='Tanh'):
        super().__init__(seq_dim, seq_len, hidden_size, n_lstm_layers, activation)
        self.gen_type = 'LSTMd'
        self.seq_dim = seq_dim # dimension of the time series e.g. how many stocks
        # self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=seq_dim, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size*2),
            activation(),
            nn.Linear(hidden_size*2, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1+129), # (1 exp rate for duration of note, 128+1 values for pitch and rest)
        )

    def _condition_lstm(self, hist_x):
        batch_size = hist_x.shape[0] # noise shape: batch_size, seq_len, noise_dim
        hist_len = hist_x.shape[1] # hist_x shape: batch_size, hist_len, seq_dim

        h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        seq = []
        if hist_x is not None: # feed in the historical data to get the hidden state
            for i in range(hist_len):
                x = hist_x[:,i:i+1,:]
                _, (h, c) = self.rnn(x, (h, c))
                seq.append(x)
        return seq, h, c

    def _generate_sequence(self, seq, h, c):
        # print(self.seq_len, len(seq), dts.shape, noise.shape)
        x = seq[-1] # (batch_size, 1, seq_dim)
        for i in range(self.seq_len-len(seq)): # iterate over the remaining time steps
            # print(x.shape, noise[:,i:i+1,:].shape, dt.shape)
            z, (h, c) = self.rnn(x, (h, c))
            z = self.output_net(z)
            seq.append(x)

            rate = z[:,0:1] # (batch_size, 1)
            pitch = z[:,1:] # (batch_size, 129)

            duration = exponential.Exponential(torch.exp(start_rate))
            end_dist = exponential.Exponential(torch.exp(end_rate))
            pitch_dist = categorical.Categorical(logits=pitch)
            velocity_dist = categorical.Categorical(logits=velocity)

            start = start_dist.rsample() + last_start # (batch_size, 1)
            end = end_dist.rsample() + start # (batch_size, 1)
            pitch = pitch_dist.sample().unsqueeze(-1) / self.scale
            velocity = velocity_dist.sample().unsqueeze(-1) / self.scale
        output_seq = torch.cat(seq, dim=1)
        return output_seq

    def forward(self, noise, t, hist_x=None, abs_path=False, abs_returns=False):
        x, noise, dts, h, c, seq = self._condition_lstm(noise, hist_x, t)
        output_seq = self._generate_sequence(x, seq, noise, dts, h, c)
        return self._return_output_seq(output_seq, None, abs_path, abs_returns)