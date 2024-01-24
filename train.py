import os
import re
import json
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sigkernel as ksig
from utils.data import *

REALTIME_MODELS = ['LSTMptime', 'LSTMCauchy', 'LSTMNF', 'LSTMsqrt', 'LSTMdetach', 'LSTMd', 'Transformer', 'T2']
LSTM_MODELS = ['LSTMptime', 'LSTMCauchy', 'LSTMNF', 'LSTMsqrt', 'LSTMdetach', 'LSTMd']
TRANSFORMER_MODELS = ['Transformer', 'T2']
PROBABILISTIC_MODELS = ['LSTMp', 'LSTMpVP', 'LSTMpdt', 'LSTMptime', 'LSTMCauchy', 'LSTMNF', 'LSTMsqrt', 'LSTMdetach']
N_LAGS = 3

def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def get_params_dicts(vars, rl=False):
    '''
    Returns data_params, model_params and train_params dictionaries from vars dictionary
    '''
    data_params = {
        'batch_size': vars['batch_size'],
        'sample_len': vars['sample_len'],
        'sample_model': vars['sample_model'], # GBM, Heston, OU, RealData, Realdt
        'seed': vars['seed'],
        'time_dim': vars['time_dim'],
    }

    model_params = {
        'signature_kernel_type': vars['signature_kernel_type'], # truncated, pde
        'static_kernel_type': vars['static_kernel_type'], # rbf, rbfmix, rq, rqmix, rqlinear for truncated / rbf, rq for pde
        'gen_type': vars['gen_type'], # MLP, LSTM, LSTMp, LSTMpdt, LSTMptime, LSTMpVP, LSTMCauchy, LSTMNF
        'hidden_size': vars['hidden_size'],
        'activation': vars['activation'], # pytorch activation function name
        'noise_dim': vars['noise_dim'],
        'seq_dim': vars['seq_dim'],
        'conditional': vars['conditional'],
    }

    train_params = {
        'epochs': vars['epochs'],
        'start_lr': vars['start_lr'],
        'patience': vars['patience'],
        'lr_factor': vars['lr_factor'],
        'early_stopping': vars['early_stopping'],
        'kernel_sigma': vars['kernel_sigma'],
        'mmd_ks_threshold': vars['mmd_ks_threshold'],
        'ks_factor': vars['ks_factor'],
        'reset_lr': vars['reset_lr'],
        'min_kernel_sigma': vars['min_kernel_sigma'],
        'mmd_stop_threshold': vars['mmd_stop_threshold'],
        'num_losses': vars['num_losses'],
    }

    if vars['dtype'] == torch.float64:
        data_params['dtype'] = 'float64'
    elif vars['dtype'] == torch.float32:
        data_params['dtype'] = 'float32'
    else:
        raise ValueError('dtype must be float32 or float64')

    if (data_params['sample_model'] == 'RealData' or
        data_params['sample_model'] == 'Realdt' or
        data_params['sample_model'] == 'SPX_rates'):

        data_params['stride'] = vars['stride']
        data_params['lead_lag'] = vars['lead_lag'] if 'lead_lag' in vars else False
        if data_params['lead_lag']:
            data_params['lags'] = vars['lags']

    if model_params['signature_kernel_type'] == 'truncated':
        model_params['n_levels'] = vars['n_levels']
    elif model_params['signature_kernel_type'] == 'pde':
        model_params['dyadic_order'] = vars['dyadic_order']
    else:
        raise ValueError('signature_kernel_type must be truncated or pde')

    if model_params['gen_type'] in LSTM_MODELS:
        model_params['n_lstm_layers'] = vars['n_lstm_layers']
    elif model_params['gen_type'] in TRANSFORMER_MODELS:
        model_params['conv_kernel_size'] = vars['conv_kernel_size']
        model_params['conv_stride'] = vars['conv_stride']
        model_params['n_channels'] = vars['n_channels']
        model_params['n_head'] = vars['n_head']
        model_params['n_transformer_layers'] = vars['n_transformer_layers']

    scramble_noise = vars['scramble_noise'] if 'scramble_noise' in vars else False
    if scramble_noise:
        model_params['scramble_noise'] = True

    if model_params['conditional'] == True:
        model_params['hist_len'] = vars['hist_len']
        model_params['include_hist'] = vars['include_hist']

    train_params['entropy_loss_coef'] = vars['entropy_loss_coef'] if 'entropy_loss_coef' in vars else 0.
    train_params['loss_type'] = vars['loss_type'] if 'loss_type' in vars else 'mmd'

    return data_params, model_params, train_params

def start_writer(data_params, model_params, train_params, rl=False, rl_params=None, env_params=None, dir=None):
    '''
    Starts a tensorboard writer and logs data, model and training parameters
    Returns the writer
    '''
    sample_model = data_params['sample_model']
    gen_type = model_params['gen_type']
    signature_kernel_type = model_params['signature_kernel_type']
    static_kernel_type = model_params['static_kernel_type']
    levels_or_order = model_params['dyadic_order'] if signature_kernel_type == 'pde' else model_params['n_levels']
    if dir is None: # fresh run from start
        if rl: # use RL comment type and add RL parameters
            algo = rl_params['algo']
            writer = SummaryWriter(comment=f'_{algo}_{signature_kernel_type}_{static_kernel_type}_{levels_or_order}')
            assert rl_params is not None and env_params is not None, 'rl_params and env_params must be provided'
            writer.add_text('RL parameters', pretty_json(rl_params))
            writer.add_text('Env parameters', pretty_json(env_params))
        else: # continue training from checkpoint
            writer = SummaryWriter(comment=f'_{sample_model}_{gen_type}_{signature_kernel_type}_{static_kernel_type}_{levels_or_order}')
        writer.add_text('Data parameters', pretty_json(data_params))
        writer.add_text('Model parameters', pretty_json(model_params))
        writer.add_text('Training parameters', pretty_json(train_params))
    else:
        writer = SummaryWriter(log_dir=dir, comment=f'_{sample_model}_{gen_type}_{signature_kernel_type}_{static_kernel_type}_{levels_or_order}')

    writer.flush()
    return writer

def get_dataloader(**kwargs):
    '''
    Returns a dataloader for the specified sample_model
    '''
    if kwargs['dtype'] == 'float32':
        dtype = torch.float32
    elif kwargs['dtype'] == 'float64':
        dtype = torch.float64
    else:
        raise ValueError('dtype must be float32 or float64')

    if dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.float64:
        np_dtype = np.float64

    sample_model = kwargs['sample_model']
    n_samples = kwargs['n_samples'] if 'n_samples' in kwargs else None
    sample_len = kwargs['sample_len']
    time_dim = kwargs['time_dim'] if 'time_dim' in kwargs else True
    stride = kwargs['stride'] if 'stride' in kwargs else None
    lead_lag = kwargs['lead_lag'] if 'lead_lag' in kwargs else False
    lags = kwargs['lags'] if 'lags' in kwargs else None
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
    seed = kwargs['seed'] if 'seed' in kwargs else None

    if sample_model == 'GBM':
        pass

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(True if sample_model == 'RealData' or sample_model == 'Realdt' else False))
    # return dataloader

def get_signature_kernel(device_ids=None, **kwargs):
    signature_kernel_type = kwargs['signature_kernel_type'] if 'signature_kernel_type' in kwargs else 'truncated'
    static_kernel_type = kwargs['static_kernel_type']
    kernel_sigma = kwargs['kernel_sigma'] if 'kernel_sigma' in kwargs else None
    n_levels = kwargs['n_levels'] if signature_kernel_type == 'truncated' else None
    # robust = kwargs['robust'] if signature_kernel_type == 'truncated' else False
    # normalization = 3 if robust else 0
    dyadic_order = kwargs['dyadic_order'] if signature_kernel_type == 'pde' else None

    if signature_kernel_type == 'truncated':
        if static_kernel_type == 'linear':
            static_kernel = ksig.static.kernels.LinearKernel()
        elif static_kernel_type == 'rbf':
            static_kernel = ksig.static.kernels.RBFKernel(sigma=kernel_sigma)
        elif static_kernel_type == 'rbfmix':
            static_kernel = ksig.static.kernels.RBFKernelMix(sigma=kernel_sigma)
        elif static_kernel_type == 'rq':
            static_kernel = ksig.static.kernels.RationalQuadraticKernel(sigma=kernel_sigma)
        elif static_kernel_type == 'rqmix':
            static_kernel = ksig.static.kernels.RationalQuadrticKernelMix(sigma=kernel_sigma)
        elif static_kernel_type == 'rqlinear':
            static_kernel = ksig.static.kernels.RQMixLinear(sigma=kernel_sigma)
        # kernel = ksig.kernels.SignatureKernel(n_levels=n_levels, order=n_levels, normalization=normalization, static_kernel=static_kernel, device_ids=device_ids)
        kernel = ksig.kernels.SignatureKernel(n_levels=n_levels, order=n_levels, normalization=0, static_kernel=static_kernel, device_ids=device_ids)
    elif signature_kernel_type == 'pde':
        if static_kernel_type == 'rbf':
            static_kernel = ksig.sigkernelpde.RBFKernel(sigma=kernel_sigma)
        elif static_kernel_type == 'rq':
            static_kernel = ksig.sigkernelpde.RationalQuadraticKernel(sigma=kernel_sigma, alpha=1.0)
        kernel = ksig.sigkernelpde.SigKernelPDE(static_kernel, dyadic_order)

    return kernel

def get_generator(**kwargs):
    gen_type = kwargs['gen_type']
    sample_len = kwargs['sample_len']
    noise_dim = kwargs['noise_dim']
    seq_dim = kwargs['seq_dim']
    hidden_size = kwargs['hidden_size']
    n_lstm_layers = kwargs['n_lstm_layers'] if 'n_lstm_layers' in kwargs else None
    activation = kwargs['activation'] if 'activation' in kwargs else None
    hist_len = kwargs['hist_len'] if 'hist_len' in kwargs else None
    conv_kernel_size = kwargs['conv_kernel_size'] if 'conv_kernel_size' in kwargs else None
    conv_stride = kwargs['conv_stride'] if 'conv_stride' in kwargs else None
    n_channels = kwargs['n_channels'] if 'n_channels' in kwargs else None
    n_head = kwargs['n_head'] if 'n_head' in kwargs else None
    n_transformer_layers = kwargs['n_transformer_layers'] if 'n_transformer_layers' in kwargs else None
    if gen_type in TRANSFORMER_MODELS:
        assert hist_len is not None, 'hist_len must be provided for Transformer'
        assert conv_kernel_size is not None, 'conv_kernel_size must be provided for Transformer'
        assert conv_stride is not None, 'conv_stride must be provided for Transformer'
        assert n_channels is not None, 'n_channels must be provided for Transformer'
        assert n_head is not None, 'n_head must be provided for Transformer'
        assert n_transformer_layers is not None, 'n_transformer_layers must be provided for Transformer'
        assert kwargs['conditional'] == True, 'conditional must be True for Transformer'
    elif gen_type in LSTM_MODELS:
        assert n_lstm_layers is not None, 'n_lstm_layers must be provided for non-Transformer models'

    if gen_type == 'LSTM':
        generator = None
    else:
        raise ValueError('gen_type unknown')
    return generator

def write_stats(X, output, sample_model, seq_dim, writer, time_dim, step, dt=None):
    '''
    Returns sample and output mean and std where mean and std are calculated on annualised returns
    Uses fixed dt for simulated data and actual dt in years for real data
    '''
    if sample_model == 'Realdt':
        dt_data = X[:,:,:1].cpu().clone().detach().numpy() if X.is_cuda else X[:,:,:1].clone().detach().numpy()
        dt_data = np.diff(dt_data, axis=1) # (batch_size, seq_len-1, 1)
    else:
        assert dt is not None, 'dt must be provided for sample_model != Realdt'

    # NOTE: (s:e) is used to exclude time dimension and the abs returns path
    s = 1 if time_dim else 0
    e = seq_dim+s
    output_cpu = output.cpu().clone().detach().numpy()[:,:,s:e] if output.is_cuda else output.clone().detach().numpy()[:,:,s:e] # (batch_size, seq_len, seq_dim)
    output_mean, output_std, output_ret_lags, output_abs_lags = get_stats(output_cpu, dt_data) if sample_model == 'Realdt' else get_stats(output_cpu, dt)
    sample_cpu = X.cpu().clone().detach().numpy()[:,:,s:e] if X.is_cuda else X.clone().detach().numpy()[:,:,s:e] # (batch_size, seq_len, seq_dim)
    sample_mean, sample_std, sample_ret_lags, sample_abs_lags = get_stats(sample_cpu, dt_data) if sample_model == 'Realdt' else get_stats(sample_cpu, dt)
    for i in range(seq_dim):
        suffix = '' if i == 0 else f'_{i}'
        writer.add_scalar(f'Stats/Output_mean{suffix}', output_mean[i], step)
        writer.add_scalar(f'Stats/Output_std{suffix}', output_std[i], step)
        writer.add_scalar(f'Stats/Sample_mean{suffix}', sample_mean[i], step)
        writer.add_scalar(f'Stats/Sample_std{suffix}', sample_std[i], step)
        for j in range(N_LAGS):
            writer.add_scalar(f'ACF_returns/Output{suffix}_lag_{j+1}', output_ret_lags[i*N_LAGS+j], step)
            writer.add_scalar(f'ACF_abs/Output{suffix}_lag_{j+1}', output_abs_lags[i*N_LAGS+j], step)
            writer.add_scalar(f'ACF_returns/Sample{suffix}_lag_{j+1}', sample_ret_lags[i*N_LAGS+j], step)
            writer.add_scalar(f'ACF_abs/Sample{suffix}_lag_{j+1}', sample_abs_lags[i*N_LAGS+j], step)

def get_stats(sample, dt):
    returns = np.diff(sample, axis=1) # (batch_size, seq_len-1, seq_dim)
    sample_mean = (returns / dt).mean(axis=(0,1))
    sample_std = (returns / np.sqrt(dt)).std(axis=(0,1))
    avg_ret_lags = []
    avg_abs_lags = []
    for i in range(returns.shape[-1]): # returns.shape[-1] = seq_dim
        for j in range(N_LAGS):
            avg_ret_lags.append(np.average([pd.Series(returns[k,:,i]).autocorr(lag=j+1) for k in range(len(returns))]))
            avg_abs_lags.append(np.average([pd.Series(np.abs(returns[k,:,i])).autocorr(lag=j+1) for k in range(len(returns))]))
    return sample_mean, sample_std, avg_ret_lags, avg_abs_lags

def train(generator, kernel, dataloader, rng, writer, device, device_ids, fig_freq=10, checkpoint=None, **kwargs):
    # set up training parameters
    epochs = kwargs['epochs']
    # batch_size = kwargs['batch_size']
    kernel_sigma = kwargs['kernel_sigma'] if checkpoint is None else checkpoint['kernel_sigma']
    min_kernel_sigma = kwargs['min_kernel_sigma']
    mmd_ks_threshold = kwargs['mmd_ks_threshold']
    ks_factor = kwargs['ks_factor']
    lr_factor = kwargs['lr_factor']
    patience = kwargs['patience']
    # early_stopping = kwargs['early_stopping']
    mmd_stop_threshold = kwargs['mmd_stop_threshold']
    reset_lr = kwargs['reset_lr']
    num_losses = kwargs['num_losses']
    lead_lag = kwargs['lead_lag'] if 'lead_lag' in kwargs else False
    lags = kwargs['lags'] if 'lags' in kwargs else None
    gen_type = kwargs['gen_type']
    start_lr = kwargs['start_lr']
    # sample_len = kwargs['sample_len']
    # seq_dim = kwargs['seq_dim']
    # noise_dim = kwargs['noise_dim']
    # scramble_noise = kwargs['scramble_noise'] if 'scramble_noise' in kwargs else False
    # sample_model = kwargs['sample_model']
    conditional = kwargs['conditional'] if 'conditional' in kwargs else False
    if conditional:
        hist_len = kwargs['hist_len']
        include_hist = kwargs['include_hist']

    if kwargs['dtype'] == 'float32':
        dtype = torch.float32
    elif kwargs['dtype'] == 'float64':
        dtype = torch.float64
    else:
        raise ValueError('dtype must be float32 or float64')

    loss_type = kwargs['loss_type'] if 'loss_type' in kwargs else 'mmd'
    entropy_loss_coef = kwargs['entropy_loss_coef'] if 'entropy_loss_coef' in kwargs else 0.
    if entropy_loss_coef > 0.:
        assert gen_type in PROBABILISTIC_MODELS, 'entropy loss can only be used with probabilistic models'

    last_k_losses = deque(maxlen=num_losses) if checkpoint is None else checkpoint['last_k_losses']
    optimizer = torch.optim.Adam(generator.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_factor, verbose=True)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = 0 if checkpoint is None else checkpoint['epoch'] + 1
    best_loss = [np.inf, 0] if checkpoint is None else checkpoint['best_loss']
    for epoch in range(start_epoch, epochs):
        losses = [] # due to legacy code, losses is actually the mmd
        type_losses = [] # used for actual loss used for backpropagation
        entropy_losses = []
        for batch_num, X in enumerate(tqdm(dataloader)):
            X = X.to(device)

            # if scramble_noise:
            #     idx = torch.randperm(noise.nelement())
            #     noise = noise.view(-1)[idx].view(noise.size())

            if (gen_type not in PROBABILISTIC_MODELS) and (gen_type not in REALTIME_MODELS):
                output = generator(noise)
            elif (gen_type in PROBABILISTIC_MODELS) and (gen_type not in REALTIME_MODELS):
                output, entropy = generator(noise)
            elif gen_type in REALTIME_MODELS:
                t = X[:,:,:1] # time dimension of path is always first series of the last dim for REALTIME_MODELS
                if conditional:
                    hist_x = X[:,:hist_len,1:] # history portion of path
                    if gen_type in PROBABILISTIC_MODELS:
                        output, entropy = generator(noise, t, hist_x=hist_x, abs_path=abs_path) # only pass time dim for generated portion as not needed for rnn
                    else:
                        output = generator(noise, t, hist_x=hist_x, abs_path=abs_path) # NOTE: abs_returns is not used at the moment
                    output = torch.cat([t, output], axis=-1) # concatenate time and history + generated path along time series value dimension
                    if not include_hist: # remove provided history portion of path if include_hist is False
                        X = X[:,hist_len:,:]
                        output = output[:,hist_len:,:]
                else:
                    if gen_type in PROBABILISTIC_MODELS:
                        output, entropy = generator(noise, t, abs_path=abs_path)
                    else:
                        output = generator(noise, t, abs_path=abs_path)
                    output = torch.cat([t, output], axis=-1) # concatenate time and generated path along time series value dimension

            # NOTE: mean/std/acf is calculated and logged to TB before all the potential transformations
            # write_stats(X if log_series else torch.log(X), output, sample_model, seq_dim, writer, time_dim, step=epoch*len(dataloader) + batch_num, dt=dt)

            # exponentiate if log_series is False but only for non-time dim NOTE: output is in log series
            if not log_series: output[:,:,1 if time_dim else 0:] = torch.exp(output[:,:,1 if time_dim else 0:])

            if lead_lag:
                X = batch_lead_lag_transform(X[:,:,1:], X[:,:,0:1], lags) # inputs are (price series, time dimension, lags to use)
                output = batch_lead_lag_transform(output[:,:,1:], output[:,:,0:1], lags)

            # compute loss
            optimizer.zero_grad()
            if loss_type == 'mmd':
                loss = ksig.tests.mmd_loss_no_compile(X, output, kernel)
                mmd = loss
            elif loss_type == 'mse_mmd':
                loss = ksig.tests.mmd_loss_no_compile(X, output, kernel)
                mmd = loss
                loss = nn.MSELoss()(loss, torch.tensor(0., device=device, dtype=dtype, requires_grad=False))
            elif loss_type == 'similiarity':
                mmd, loss = ksig.tests.similarity_loss_no_compile(X, output, kernel)
            elif loss_type == 'score':
                mmd, loss = ksig.tests.scoring_rule_no_compile(X, output, kernel)
            elif loss_type == 'mse_score':
                mmd, loss = ksig.tests.scoring_rule_no_compile(X, output, kernel)
                loss = nn.MSELoss()(loss, torch.tensor(0., device=device, dtype=dtype, requires_grad=False))
            else:
                raise ValueError('loss_type not recognized')

            losses.append(mmd.item()) # due to legacy code, losses is actually the mmd
            type_losses.append(loss.item())

            # entropy loss is added for backpropagation
            if entropy_loss_coef > 0.:
                entropy_loss = -entropy_loss_coef * entropy.mean()
                entropy_losses.append(entropy_loss.item())
                loss = loss + entropy_loss

            # backpropagate and update weights
            loss.backward()
            optimizer.step()

            # log batch loss, sample/output mean, std where mean and std are calculated on annualised returns and autocorrelation coeffs
            # writer.add_scalar('Loss/Batch', mmd.item(), epoch*len(dataloader) + batch_num)
            # if gen_type in PROBABILISTIC_MODELS:
            #     writer.add_scalar('Stats/Gen_std', generator.stds.detach().cpu().mean().item(), epoch*len(dataloader) + batch_num)
            #     writer.add_scalar('Stats/Gen_std_std', generator.stds.detach().cpu().std().item(), epoch*len(dataloader) + batch_num)
            #     writer.add_scalar('Stats/Gen_mean', generator.mus.detach().cpu().mean().item(), epoch*len(dataloader) + batch_num)
            #     writer.add_scalar('Stats/Gen_mean_std', generator.mus.detach().cpu().std().item(), epoch*len(dataloader) + batch_num)

        # log epoch loss and plot generated samples
        epoch_mmd = np.average(losses) # average batch mmd for epoch
        epoch_loss = np.average(type_losses) # average batch losses for epoch
        last_k_losses.append(epoch_loss)
        avg_k_losses = np.average(last_k_losses)
        scheduler.step(avg_k_losses)
        # writer.add_scalar('Loss/Epoch', epoch_mmd, epoch) # NOTE: loss recorded for tensorboard is always avg mmd of epoch
        # writer.add_scalar(f'Loss/Epoch_{loss_type}', epoch_loss, epoch)
        # writer.add_scalar('Loss/Epoch_avg_k', avg_k_losses, epoch)
        # writer.add_scalar('Param/LR', optimizer.param_groups[0]['lr'], epoch)
        print(f'Epoch {epoch}, mmd: {epoch_mmd}, loss: {epoch_loss}, avg_last_{num_losses}_loss: {avg_k_losses}, kernel_sigma: {kernel_sigma}')

        # save model if avg_k_losses is the best loss so far and kernel_sigma is at the minimum
        # if avg_k_losses < best_loss[0] and kernel_sigma == min_kernel_sigma:
        #     best_loss = [avg_k_losses, epoch]
        #     print(f'Saving model at epoch {epoch}')
        #     torch.save(generator.state_dict(), f'./{writer.log_dir}/best_model_{kernel_sigma:3f}ls.pt')
        # elif epoch - best_loss[1] >= early_stopping:
        #     print(f'Early stopping at epoch {epoch}')
        #     break

        # if entropy_loss_coef > 0.:
        #     writer.add_scalar('Loss/Entropy', np.average(entropy_losses), epoch)

        # if epoch % fig_freq == 0 or epoch == epochs-1:
        #     for i in range(seq_dim):
        #         suffix = '' if i == 0 else f'_{i}'
        #         start = 1 if time_dim else 0 # exclude time dimension
        #         end = -max(lags) if lead_lag else sample_len # lead-lag transformation will add max(lags) to the end of the path
        #         plot_x = X[:,:end,start+i].cpu().clone().detach().numpy().T
        #         plt.plot(np.exp(plot_x) if log_series else plot_x);
        #         writer.add_figure(f'Generated vs reference samples/Reference_{suffix}', plt.gcf(), epoch)
        #         plot_x = output[:,:end,start+i].cpu().clone().detach().numpy().T
        #         plt.plot(np.exp(plot_x) if log_series else plot_x);
        #         writer.add_figure(f'Generated vs reference samples/Generated_{suffix}', plt.gcf(), epoch)

        # stop training if MMD goes below mmd_stop_threshold
        if kernel_sigma == min_kernel_sigma and len(last_k_losses) == num_losses and np.max(last_k_losses) < mmd_stop_threshold:
            break

        # adjust kernel_sigma and reset if MMD goes below mmd_ks_threshold
        if epoch_mmd < mmd_ks_threshold and kernel_sigma > min_kernel_sigma:

            # reset best loss for early stopping
            best_loss = [np.inf, epoch]

            # reset optimizer if reset_lr is not None but always reset scheduler
            if reset_lr is not None:
                optimizer = torch.optim.Adam(generator.parameters(), lr=reset_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_factor, verbose=True)

            # update kernel kernel_sigma
            kernel_sigma *= ks_factor
            kernel_sigma = max(kernel_sigma, min_kernel_sigma)
            kwargs['kernel_sigma'] = kernel_sigma
            kernel = get_signature_kernel(device_ids, **kwargs)
            print(f'New kernel_sigma: {kernel_sigma} at epoch {epoch}')
        # writer.add_scalar('Param/kernel_sigma', kernel_sigma, epoch)

        # writer.flush()

        # save weights and random states for continued training
        # torch.save({'epoch': epoch,
        #             'rng_state': rng.bit_generator.state,
        #             'generator_state_dict': generator.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict(),
        #             'torch_rng_state': torch.get_rng_state(),
        #             'best_loss': best_loss,
        #             'kernel_sigma': kernel_sigma,
        #             'last_k_losses': last_k_losses,
        #             'dataset_rng_state': dataloader.dataset.rs.get_state() if garch else None
        #             }, f'./{writer.log_dir}/checkpoint.pt')

    # torch.save(generator.state_dict(), f'./{writer.log_dir}/generator.pt')
