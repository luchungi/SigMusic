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
        Transformer encoder output is then fed into a linear layer to produce the output sequence
        Noise is concatenated with output of the transformer encoder and then fed into a MLP to produce the output
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
            nn.Linear(hidden_size, 1), # (exp rate for duration, max_pitch+1 values for pitch and rest)
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
            pitch = pitch_dist.sample().unsqueeze(-1) * self.scale

            note = x = torch.cat([duration, pitch], dim=1).unsqueeze(1) #(batch_size, 1, 2)
            seq.append(note)

            hist_x = hist_x.roll(-1, dims=2)
            hist_x[:,:,-1:] = note.permute(0,2,1)

        return torch.cat(seq, dim=1) # (batch_size, seq_len, 4)

    def forward(self, x):
        return self._generate_sequence(x)

class TransInc(nn.Module):
    def __init__(self, noise_dim:int, seq_dim:int, seq_len:int, hist_len:int, range:int,
                 kernel_size:int|list[int], stride:int|list[int], n_channels:int,
                 n_head:int, n_transformer_layers:int, hidden_size:int,
                 activation:str):
        '''
        Transformer architecture with a deterministic output
        Uses a conv1d layer to encode the input sequence into an embedding
        Embedding is then fed into a transformer encoder
        Transformer encoder output is then fed into a linear layer to produce the output sequence
        '''
        super().__init__()
        self.gen_type = 'TransInc'
        self.noise_dim = noise_dim
        self.seq_dim = seq_dim # dimension of the time series e.g. how many stocks
        self.seq_len = seq_len # length of the time series to generate i.e. does not include the historical data NOTE: different to LSTMRealdt
        self.hist_len = hist_len # length of historical path to feed to network (length of returns = hist_len-1)
        self.range = range
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.stride = stride
        self.n_head = n_head
        self.n_transformer_layers = n_transformer_layers
        self.hidden_size = hidden_size
        activation = getattr(nn, activation)

        self.encoded_seq_len = int((((hist_len) - kernel_size) / stride + 1))
        self.encoder = nn.Conv1d(seq_dim+noise_dim, n_channels, kernel_size=kernel_size, stride=stride) # +1 for noise dim
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=n_channels, nhead=n_head, dim_feedforward=hidden_size,
                                                                    dropout=0., batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, n_transformer_layers)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoded_seq_len * n_channels, hidden_size*2),
            activation(),
            nn.Linear(hidden_size*2, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1), # (exp rate for duration, max_pitch+1 values for pitch and rest)
            nn.Tanh()
        )

    def _generate_sequence(self, x, noise): # (batch_size, sample_len, seq_dim))
        seq = []
        hist_x = x[:,:self.hist_len,:].permute(0,2,1) # (batch_size, seq_dim, hist_len)
        hist_noise = noise[:,:self.hist_len,:].permute(0,2,1) # (batch_size, noise_dim, hist_len)

        for _ in range(self.seq_len-self.hist_len):
            input = torch.cat([hist_x, hist_noise], dim=1)
            z = self.encoder(input).permute(0,2,1) # permute back to (batch_size, encoded_seq_len, n_channels)
            # print(z.shape)
            z = self.transformer_encoder(z) # (batch_size, encoded_seq_len, n_channels)
            z = self.decoder(z) # (batch_size, 1)
            dpitch = (z * self.range) # (batch_size, 1, 1)
            # print(dpitch.shape)
            seq.append(dpitch.unsqueeze(-1))

            hist_x = hist_x.roll(-1, dims=2)
            hist_x[:,:,-1] = dpitch

        dpitch = torch.cat(seq, dim=1)
        # print(dpitch.shape)
        dpitch = torch.cat([x[:,:self.hist_len,-1:], dpitch], dim=1)
        # print(dpitch.shape)
        seq = torch.cat([x[:,:,:-1], dpitch], dim=-1)
        # print(seq.shape)

        return seq

    def forward(self, x, noise):
        return self._generate_sequence(x, noise)

class LSTMusic(nn.Module):
    def __init__(self, seq_dim: int, seq_len: int, max_pitch: int, hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh'):
        super().__init__()
        self.gen_type = 'LSTMusic'
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
    '''
    LSTM model that generates a sequence of delta pitch values
    The output of the LSTM is passed through a linear layer followed by a tanh activation to ensure that the delta pitch values are within the range [-1, 1]
    This value is then multiplied by the dpitch_range (input) to get the actual delta pitch value
    The gap and pitch duration are given as input from the reference sample
    Previous solution tried to generate these values by taking the exponential of the output of the LSTM
    However, the gap ended getting too large towards the end of the sequence
    '''

    def __init__(self, noise_dim:int, seq_dim: int, seq_len: int,
                 dpitch_range: int=24, scale: float=1.0,
                 hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh'):
        super().__init__()
        self.gen_type = 'LSTMinc'
        self.seq_dim = seq_dim # dimension of the time series
        self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series
        self.dpitch_range = dpitch_range
        self.scale = scale
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=noise_dim + seq_dim, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        # self.output_net = nn.Sequential(
        #     nn.Linear(hidden_size, 27), # 1st value as gap duration, 2nd as pitch duration and the next 25 values as logits for delta pitch distribution
        # )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1), # delta pitch value
            nn.Tanh()
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

    def _generate_sequence(self, output, noise, h, c, gap_duration):
        gen_seq = []
        for i in range(noise.shape[1]+1): # +1 for the first note which is using the output passed in
            z = self.output_net(output)
            # gap_duration = torch.exp(z[:,:,:2]) # ensure that the duration and pause duration are positive
            deltapitch = z * self.dpitch_range * self.scale
            x = torch.cat([gap_duration[:,i:i+1,:], deltapitch], dim=-1)
            gen_seq.append(x)
            if i < noise.shape[1]:
                input = torch.cat([x, noise[:,i:i+1,:]], dim=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                output, (h, c) = self.rnn(input, (h, c))
        output_seq = torch.cat(gen_seq, dim=1)
        return output_seq

    def forward(self, noise, hist_x=None, gap_duration=None):
        output, noise, h, c = self._condition_lstm(noise, hist_x)
        output_seq = self._generate_sequence(output, noise, h, c, gap_duration)
        if hist_x is None:
            return output_seq
        else:
            return torch.cat([hist_x, output_seq], dim=1)

class LSTMinc_v2(nn.Module):
    '''
    LSTM model that generates a sequence of delta pitch values
    The output of the LSTM is passed through a linear layer followed by a tanh activation to ensure that the delta pitch values are within the range [-1, 1]
    This value is then multiplied by the dpitch_range (input) to get the actual delta pitch value
    The gap and pitch duration are given as input from the reference sample
    Previous solution tried to generate these values by taking the exponential of the output of the LSTM
    However, the gap ended getting too large towards the end of the sequence
    '''

    def __init__(self, noise_dim:int, seq_dim: int, seq_len: int,
                 dpitch_range: int=24, scale: float=1.0,
                 hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh'):
        super().__init__()
        self.gen_type = 'LSTMinc_v2'
        self.seq_dim = seq_dim # dimension of the time series
        self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series
        self.dpitch_range = dpitch_range
        self.scale = scale
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=noise_dim + seq_dim + 1, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        # +1 for distance from the initial note
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1), # delta pitch value
            nn.Tanh()
        )

    def _condition_lstm(self, noise, hist_x):
        batch_size = noise.shape[0] # noise shape: batch_size, seq_len, noise_dim
        h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        dist = torch.cumsum(hist_x[:,:,-1:], dim=1) # distance from the initial note
        # print(hist_x[0,:,-1], dist[0,:,-1])

        if hist_x is not None: # feed in the historical data to get the hidden state
            input = torch.cat([hist_x, dist, noise[:, :hist_x.shape[1], :]], dim=-1)
            output, (h, c) = self.rnn(input, (h, c))
            noise = noise[:,hist_x.shape[1]:,:] # set the noise to start from the end of the historical data
        else:
            output = torch.zeros(batch_size, 1, self.hidden_size, requires_grad=False, device=noise.device)
        return output[:,-1:,:], noise, h, c, dist[:,-1:,:]

    def _generate_sequence(self, output, noise, h, c, gap_duration, dist):
        gen_seq = []
        for i in range(noise.shape[1]+1): # +1 for the first note which is using the output passed in
            z = self.output_net(output)
            # gap_duration = torch.exp(z[:,:,:2]) # ensure that the duration and pause duration are positive
            deltapitch = z * self.dpitch_range * self.scale
            x = torch.cat([gap_duration[:,i:i+1,:], deltapitch], dim=-1)
            gen_seq.append(x)
            dist = dist + deltapitch
            if i < noise.shape[1]:
                input = torch.cat([x, dist, noise[:,i:i+1,:]], dim=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                output, (h, c) = self.rnn(input, (h, c))
        output_seq = torch.cat(gen_seq, dim=1)
        return output_seq

    def forward(self, noise, hist_x=None, gap_duration=None):
        output, noise, h, c, dist = self._condition_lstm(noise, hist_x)
        output_seq = self._generate_sequence(output, noise, h, c, gap_duration, dist)
        if hist_x is None:
            return output_seq
        else:
            return torch.cat([hist_x, output_seq], dim=1)

class LSTMgate(nn.Module):
    '''
    LSTM model that generates a sequence of delta pitch values
    The output of the LSTM is passed through a linear layer followed by a tanh activation to ensure that the delta pitch values are within the range [-1, 1]
    This value is then multiplied by the dpitch_range (input) to get the actual delta pitch value
    The gap and pitch duration are given as input from the reference sample
    Previous solution tried to generate these values by taking the exponential of the output of the LSTM
    However, the gap ended getting too large towards the end of the sequence
    '''

    def __init__(self, noise_dim:int, seq_dim: int, seq_len: int, hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh', dpitch_range: int=24):
        super().__init__()
        self.gen_type = 'LSTMgate'
        self.seq_dim = seq_dim # dimension of the time series
        self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.dpitch_range = dpitch_range

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=noise_dim + seq_dim + 1, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.cluster_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1), # delta pitch value
            nn.Tanh()
        )

    def _condition_lstm(self, noise, hist_x, cluster):
        batch_size = noise.shape[0] # noise shape: batch_size, seq_len, noise_dim
        h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)

        if hist_x is not None: # feed in the historical data to get the hidden state
            # input = torch.cat([hist_x, noise[:, :hist_x.shape[1], :]], dim=-1)
            input = torch.cat([hist_x, noise[:, :hist_x.shape[1], :], cluster.repeat(1,hist_x.shape[1],1)], dim=-1)
            output, (h, c) = self.rnn(input, (h, c))
            noise = noise[:,hist_x.shape[1]:,:] # set the noise to start from the end of the historical data
        else:
            output = torch.zeros(batch_size, 1, self.hidden_size, requires_grad=False, device=noise.device)
        # track_features = self.feature_net(output[:,-1:,:])
        # print(track_features.shape, cluster.shape)
        # output = track_features * cluster
        # return output, noise, h, c
        return output[:,-1:,:], noise, h, c

    def _generate_sequence(self, output, noise, h, c, gap_duration, cluster):
        gen_seq = []
        for i in range(noise.shape[1]+1): # +1 for the first note which is using the output passed in
            z = self.output_net(output)
            # gap_duration = torch.exp(z[:,:,:2]) # ensure that the duration and pause duration are positive
            deltapitch = z * self.dpitch_range
            # print(deltapitch.shape, gap_duration.shape)
            x = torch.cat([gap_duration[:,i:i+1,:], deltapitch], dim=-1)
            gen_seq.append(x)
            if i < noise.shape[1]:
                # input = torch.cat([x, noise[:,i:i+1,:]], dim=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                input = torch.cat([x, noise[:,i:i+1,:], cluster], dim=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                output, (h, c) = self.rnn(input, (h, c))
                # track_features = self.feature_net(output)
                # output = track_features * features
        output_seq = torch.cat(gen_seq, dim=1)
        return output_seq

    def forward(self, noise, cluster, hist_x=None, gap_duration=None):
        # cluster = self.cluster_net(cluster).unsqueeze(1)
        if cluster.ndim == 2:
            cluster = cluster.unsqueeze(1)
        output, noise, h, c = self._condition_lstm(noise, hist_x, cluster)
        output_seq = self._generate_sequence(output, noise, h, c, gap_duration, cluster)
        if hist_x is None:
            return output_seq
        else:
            return torch.cat([hist_x, output_seq], dim=1)