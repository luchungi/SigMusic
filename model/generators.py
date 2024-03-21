import torch
import torch.nn as nn
from torch.distributions import categorical, exponential

class TransformerMusic(nn.Module):
    def __init__(self, seq_dim:int, seq_len:int, max_pitch: int, hist_len:int, scale:float,
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
        self.max_pitch = max_pitch
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
            nn.Linear(hidden_size, 1+self.max_pitch+1), # (exp rate for duration, max_pitch+1 values for pitch and rest)
        )

    def _generate_sequence(self, x): # (batch_size, sample_len, seq_dim))
        seq = []
        hist_x = x[:,:self.hist_len,:].permute(0,2,1) # (batch_size, seq_dim, hist_len)

        for _ in range(self.seq_len-self.hist_len):
            z = self.encoder(hist_x).permute(0,2,1) # permute back to (batch_size, encoded_seq_len, n_channels)
            z = self.transformer_encoder(z) # (batch_size, encoded_seq_len, n_channels)
            z = self.decoder(z) # (batch_size, 1+1+128+128)

            rate = z[:,0:1] # (batch_size, 1)
            pitch = z[:,1:] # (batch_size, max_pitch+1)

            duration_dist = exponential.Exponential(torch.exp(rate))
            pitch_dist = categorical.Categorical(logits=pitch)

            duration = duration_dist.rsample() # (batch_size, 1)
            pitch = pitch_dist.sample().unsqueeze(-1) / self.scale

            note = x = torch.cat([duration, pitch], dim=1).unsqueeze(1) #(batch_size, 1, 2)
            seq.append(note)

            hist_x = hist_x.roll(-1, dims=2)
            hist_x[:,:,-1:] = note.permute(0,2,1)

        return torch.cat(seq, dim=1) # (batch_size, seq_len, 4)

    def forward(self, x):
        return self._generate_sequence(x)

# class LSTMusic(nn.Module):
#     def __init__(self, noise_dim: int, seq_dim: int, seq_len: int, max_pitch: int, hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh'):
#         super().__init__()
#         self.gen_type = 'LSTMd'
#         self.seq_dim = seq_dim # dimension of the time series
#         self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
#         self.seq_len = seq_len # length of the time series
#         self.hidden_size = hidden_size
#         self.n_lstm_layers = n_lstm_layers
#         self.max_pitch = max_pitch

#         activation = getattr(nn, activation)
#         self.rnn = nn.LSTM(input_size=seq_dim+noise_dim, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
#         self.output_net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(hidden_size, hidden_size*2),
#             activation(),
#             nn.Linear(hidden_size*2, hidden_size),
#             activation(),
#             nn.Linear(hidden_size, 2), # (1 for note duration, 1 for pitch)
#         )

#     def _generate_sequence(self, noise, hist_x):
#         batch_size = hist_x.shape[0] # noise shape: batch_size, seq_len, noise_dim
#         hist_len = hist_x.shape[1]
#         device = hist_x.device

#         h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=device)
#         c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=device)
#         input = torch.cat([hist_x[:,:-1,:], noise[:,:hist_len-1,:]], dim=-1) # (batch_size, hist_len-1, seq_dim+noise_dim)
#         z, (h, c) = self.rnn(input, (h, c))
#         input = torch.cat([hist_x[:,-1:,:], noise[:,hist_len-1:hist_len,:]], dim=-1) # (batch_size, 1, seq_dim+noise_dim)
#         seq = []
#         for i in range(self.seq_len-hist_len): # iterate over the remaining time steps
#             z, (h, c) = self.rnn(input, (h, c)) # (batch_size, 1, hidden_size)
#             z = self.output_net(z) # (batch_size, 1, hidden) -> (batch_size, 2)

#             duration = torch.sigmoid(z[:, 0:1]) # (batch_size, 1)
#             pitch = torch.round(torch.sigmoid(z[:, 1:2]) * self.max_pitch) # (batch_size, 1)

#             x = torch.cat([duration, pitch], dim=1).unsqueeze(1) #(batch_size, 1, 2)
#             seq.append(x)
#             input = torch.cat([x, noise[:,hist_len+i:hist_len+i+1,:]], dim=-1)

#         output_seq = torch.cat(seq, dim=1)
#         return output_seq

#     def forward(self, noise, hist_x=None):
#         return self._generate_sequence(noise, hist_x)

class LSTMusic(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, max_pitch: int, hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh'):
        super().__init__()
        self.gen_type = 'LSTMd'
        self.seq_dim = seq_dim # dimension of the time series
        # self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.max_pitch = max_pitch

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=seq_dim, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size*2),
            activation(),
            nn.Linear(hidden_size*2, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1+max_pitch+1), # (1 exp rate for duration of note, 128+1 values for pitch and rest)
        )

    def _generate_sequence(self, hist_x):
        batch_size = hist_x.shape[0] # noise shape: batch_size, seq_len, noise_dim
        hist_len = hist_x.shape[1]
        device = hist_x.device

        h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=device)
        c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=device)
        z, (h, c) = self.rnn(hist_x[:,:-1,:], (h, c))
        x = hist_x[:,-1:,:] # (batch_size, hist_len-1, seq_dim)
        seq = []
        for _ in range(self.seq_len-hist_len): # iterate over the remaining time steps
            z, (h, c) = self.rnn(x, (h, c)) # (batch_size, 1, hidden_size)
            z = self.output_net(z) # (batch_size, 1, hidden) -> (batch_size, 1, 1+max_pitch+1)

            rate = z[:,0:1] # (batch_size, 1)
            pitch = z[:,1:] # (batch_size, max_pitch+1)

            dur_dist = exponential.Exponential(torch.exp(rate))
            pitch_dist = categorical.Categorical(logits=pitch)

            duration = dur_dist.rsample() # (batch_size, 1)
            # duration = torch.sigmoid(rate)
            pitch = pitch_dist.sample().unsqueeze(-1) # (batch_size, 1)

            x = torch.cat([duration, pitch], dim=1).unsqueeze(1) #(batch_size, 1, 2)
            seq.append(x)

        output_seq = torch.cat(seq, dim=1)
        return output_seq

    def forward(self, hist_x=None):
        return self._generate_sequence(hist_x)

class LSTMinc(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, max_pitch: int, hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh'):
        super().__init__()
        self.gen_type = 'LSTMd'
        self.seq_dim = seq_dim # dimension of the time series
        # self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=seq_dim+1, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, 3), # (1 for duration to pause before starting note, 1 for note duration, 1 for pitch)
        )

    def _condition_lstm(self, noise, hist_x):
        batch_size = noise.shape[0] # noise shape: batch_size, seq_len, noise_dim
        h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)

        if hist_x is not None: # feed in the historical data to get the hidden state
            input = torch.cat([hist_x, noise[:, :hist_x.shape[1], :]], dim=-1)
            output, (h, c) = self.rnn(input, (h, c))
            noise = noise[:,hist_x.shape[1]:,:] # set the noise to start from the end of the historical data
        else:
            output = torch.zeros(batch_size, 1, self.hidden_size, requires_grad=False, device=noise.device)
        return output[:,-1:,:], noise, h, c

    def _generate_sequence(self, seq, output, noise, dts, h, c):
        gen_seq = []
        for i in range(self.seq_len-seq.shape[1]): # iterate over the remaining time steps
            x = self.output_net(output)
            x[:,:,:2] = torch.nn.ReLU()(x[:,:,:2]) # ensure that the duration and pause duration are positive
            x[:,:,2:] = torch.round(x[:,:,2:]) # round the delta pitch to the nearest integer
            gen_seq.append(x)
            if i < noise.shape[1]:
                input = torch.cat([x, noise[:,i:i+1,:], dts[:,i:i+1,:]], dim=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                output, (h, c) = self.rnn(input, (h, c))
        # print(f'Historical sequence shape: {seq.shape}')
        # print(seq)
        output_seq = torch.cat(gen_seq, dim=1)
        # print(f'Generated sequence shape: {output_seq.shape}')
        # print(output_seq)
        return output_seq

    def forward(self, noise, hist_x=None):
        x, noise, h, c = self._condition_lstm(noise, hist_x)
        output_seq = self._generate_sequence(x, noise, h, c, hist_x)
        if hist_x is None:
            return output_seq
        else:
            return torch.cat([hist_x, output_seq], dim=1)