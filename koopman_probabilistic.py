#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Alex Mallen (atmallen@uw.edu)
Built on code from Henning Lange (helange@uw.edu)
"""

import torch

from torch import nn
from torch import optim

import numpy as np

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt


class KoopmanProb(nn.Module):
    r'''

    model_obj: an object that specifies the function f and how to optimize
               it. The object needs to implement numerous function. See
               below for some examples.

    sample_num: number of samples from temporally local loss used to
                reconstruct the global error surface.

    batch_size: Number of temporal snapshots processed by SGD at a time
                default = 32
                type: int

    parallel_batch_size: Number of temporaly local losses sampled in parallel.
                         This number should be as high as possible but low enough
                         to not cause memory issues.
                         default = 1000
                         type: int

    device: The device on which the computations are carried out.
            Example: cpu, cuda:0, or list of GPUs for multi-GPU usage, i.e. ['cuda:0', 'cuda:1']
            default = 'cpu'

    seed: The seed to set for pyTorch and numpy--WARNING: does not seem to make results reproducible

    num_fourier_modes: the number of frequencies to set using the argmax values of the fourier transform. these are
                       shared between all parameters
                       condition: num_fourier_modes <= min(num_freqs)
                       default = 0

    '''

    def __init__(self, model_obj, sample_num=12, seed=None, **kwargs):

        super(KoopmanProb, self).__init__()
        self.total_freqs = model_obj.total_freqs
        self.num_freqs = model_obj.num_freqs

        if seed is not None:
            torch.set_deterministic(True)
            torch.manual_seed(seed)
            np.random.seed(seed)

        if 'device' in kwargs:
            self.device = kwargs['device']
            if type(kwargs['device']) == list:
                self.device = kwargs['device'][0]
                multi_gpu = True
            else:
                multi_gpu = False
        else:
            self.device = 'cpu'
            multi_gpu = False

        self.multi_gpu = multi_gpu

        self.parallel_batch_size = kwargs['parallel_batch_size'] if 'parallel_batch_size' in kwargs else 1000
        self.num_fourier_modes = kwargs['num_fourier_modes'] if 'num_fourier_modes' in kwargs else 0
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32

        # Initial guesses for frequencies
        self.omegas = torch.linspace(0.01, 0.5, self.total_freqs, device=self.device)

        model_obj = model_obj.to(self.device)
        self.model_obj = nn.DataParallel(model_obj, device_ids=kwargs['device']) if multi_gpu else model_obj
        self.sample_num = sample_num

    def find_fourier_omegas(self, xt, hard_code=None):
        """
        computes the fft of the data to "hard-code" self.num_fourier_modes values of omega that
        will remain constant through optimization

        :param xt: the data to initialize fourier modes with
        :param hard_code: specifically define the periods you wish to preset the model with in a list
                          pre condition: len(hard_code) == self.num_fourier_modes
        :return: omegas found
        """
        best_omegas = None
        if hard_code is not None:
            best_omegas = 2 * np.pi / torch.tensor(hard_code)
        
        elif self.num_fourier_modes > 0:
            xt_ft = np.fft.fft(np.reshape(xt, xt.size))
            adj_xt_ft = abs(xt_ft) + abs(np.flip(xt_ft))
            freqs = np.fft.fftfreq(len(xt_ft))

            best_omegas = np.zeros(self.num_fourier_modes)
            i = 0
            num_found = 0
            while num_found < self.num_fourier_modes:
                # TODO also implement partitions in fft
                amax = np.argpartition(-adj_xt_ft[:len(xt_ft) // 2], i)[i]  # ith biggest freq
                if freqs[amax] != 0 and all(abs(1 - best_omegas / freqs[amax]) > 0.1):
                    best_omegas[num_found] = freqs[amax]
                    num_found += 1
                i += 1

            best_omegas = 2 * np.pi * torch.tensor(best_omegas)
            print("fourier periods:", 2 * np.pi / best_omegas)

        idx = 0
        for num_freqs in self.num_freqs:
            self.omegas[idx:idx + self.num_fourier_modes] = best_omegas
            idx += num_freqs
        return best_omegas

    def sample_error(self, xt, which):
        '''

        sample_error computes all temporally local losses within the first
        period, i.e. between [0,2pi/t]

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        which : TYPE int
            Index of the entry of omega

        Returns
        -------
        TYPE numpy.array
            Matrix that contains temporally local losses between [0,2pi/t]
            dimensions: [T, sample_num]

        '''

        if type(xt) == np.ndarray:
            xt = torch.tensor(xt, device=self.device)

        num_samples = self.sample_num
        omega = self.omegas
        batch = self.parallel_batch_size

        t = torch.arange(xt.shape[0], device=self.device) + 1
        errors = []

        pi_block = torch.zeros((num_samples, len(omega)))
        pi_block[:, which] = torch.arange(0, num_samples) * np.pi * 2 / num_samples

        if t.shape[0] < batch:
            batch = t.shape[0]
        for i in range(int(np.ceil(xt.shape[0] / batch))):
            t_batch = t[i * batch:(i + 1) * batch][:, None]
            wt = t_batch * omega[None]
            wt[:, which] = 0
            wt = wt[:, None] + pi_block[None]
            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)
            loss = self.model_obj(k, xt[i * batch:(i + 1) * batch, None]).cpu().detach().numpy()
            errors.append(loss)

        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        return np.concatenate(errors, axis=0)

    def reconstruct(self, errors, use_heuristic=True):
        """
        reconstructs the total error surface from samples of temporally local loss functions
        :param errors: the temporally local loss functions for t=1,2... sampled within 2pi/t
        :param use_heuristic: whether to implement the unknown phase problem heuristic to improve optimization
        :return: Global loss function (the first period--from 0 to 2pi) with respect to omega, its fft
        """

        e_fft = np.fft.fft(errors)
        E_ft = np.zeros(errors.shape[0] * self.sample_num, dtype=np.complex64)

        for t in range(1, e_fft.shape[0] + 1):
            E_ft[np.arange(self.sample_num // 2) * t] += e_fft[t - 1, :self.sample_num // 2]

        E_ft = np.concatenate([E_ft, np.conj(np.flip(E_ft))])[:-1]
        E = np.real(np.fft.ifft(E_ft))

        if use_heuristic:
            E = -np.abs(E - np.median(E))
            # E = gaussian_filter(E, 5)

        return E, E_ft

    def fft(self, xt, i, verbose=False):
        '''

        fft first samples all temporaly local losses within the first period
        and then reconstructs the global error surface w.r.t. omega_i
        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        i : TYPE int
            Index of the entry of omega
        verbose : TYPE boolean, optional
            DESCRIPTION. The default is False.
        Returns
        -------
        E : TYPE numpy.array
            Global loss surface in time domain.
        E_ft : TYPE
            Global loss surface in frequency domain.
        '''

        E, E_ft = self.reconstruct(self.sample_error(xt, i))
        omegas = np.linspace(0, 0.5, len(E))

        # get the values of omega that have already been used
        omegas_current = self.omegas.cpu().detach().numpy()
        omegas_current[i] = -1
        for j, num_freqs in enumerate(self.num_freqs):
            lower = sum(self.num_freqs[:j])
            upper = sum(self.num_freqs[:j + 1])
            if lower <= i < upper:
                omegas_current = omegas_current[lower:upper]

        found = False
        j = 0
        while not found:

            amax = np.argpartition(E, j)[j]  # jth biggest freq
            # The if statement avoids non-unique entries in omega and that the
            # frequencies are 0 (should be handled by bias term)
            # "nonzero AND has a period that's more than 1 different from those that have already been discovered"
            if amax >= 1 and np.all(np.abs(2 * np.pi / omegas_current - 1 / omegas[amax]) > 1):
                found = True
                if verbose:
                    print('Setting', i, 'to', 1 / omegas[amax])

                self.omegas[i] = torch.from_numpy(np.array([omegas[amax]]))
                self.omegas[i] *= 2 * np.pi

            j += 1


        # TODO remove plotting
        # plt.plot(errs[-1])
        # plt.show()

        # plt.plot(omegas, E)
        # plt.title(f"omega {i}")
        # plt.xlabel("frequency (periods per time)")
        # plt.ylabel("loss")
        # plt.show()

        return E, E_ft

    def sgd(self, xt, iteration, weight_decay=0, verbose=False, lr_theta=1e-5, lr_omega=1e-5):
        '''

        sgd performs a single epoch of stochastic gradient descent on parameters
        of f (Theta) and frequencies omega

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        verbose : TYPE boolean, optionaly
            The default is False.

        Returns
        -------
        TYPE float
            Loss.

        '''

        batch_size = self.batch_size

        T = xt.shape[0]

        omega = nn.Parameter(self.omegas)

        # opt = optim.Adam(self.model_obj.parameters(), lr=1e-4 * (1 / (1 + np.exp(-(iteration - 15)))), betas=(0.99, 0.9999), eps=1e-5, weight_decay=weight_decay)
        opt = optim.SGD(self.model_obj.parameters(), lr=lr_theta * (1 / (1 + np.exp(-(iteration - 15)))), weight_decay=weight_decay)
        opt_omega = optim.SGD([omega], lr=lr_omega / T * (1 / (1 + np.exp(-(iteration - 15)))))

        T = xt.shape[0]
        t = torch.arange(T, device=self.device)

        losses = []

       
        for i in range(len(t) // batch_size + 1):
            if  i == len(t) // batch_size:  # remainder data with batch smaller than batch_size
                ts = t[-(len(t) % batch_size):]
            else:
                ts = t[i * batch_size:(i + 1) * batch_size]
            o = torch.unsqueeze(omega, 0)
            ts_ = torch.unsqueeze(ts, -1).type(torch.get_default_dtype()) + 1

            xt_t = torch.tensor(xt[ts.cpu().numpy(), :], device=self.device)

            wt = ts_ * o

            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)
            loss = torch.mean(self.model_obj(k, xt_t))

            if loss > 10e9:
                print("loss at:", i, loss)

            opt.zero_grad()
            opt_omega.zero_grad()

            loss.backward()

            opt.step()
            opt_omega.step()

            losses.append(loss.cpu().detach().numpy())

        if verbose:
            print('Setting periods to', 2 * np.pi / omega)

        self.omegas = omega.data

        return np.mean(losses)

    def fit(self, xt, iterations=10, interval=5, cutoff=np.inf, weight_decay=0, verbose=False, lr_theta=1e-5, lr_omega=1e-5):
        '''
        Given a dataset, this function alternatingly optimizes omega and
        parameters of f. Specifically, the algorithm performs interval many
        epochs, then updates all entries in omega. This process is repeated
        until iterations-many epochs have been performed

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        iterations : TYPE int, optional
            Total number of SGD epochs. The default is 10.
        interval : TYPE, optional
            The interval at which omegas are updated, i.e. if
            interval is 5, then omegas are updated every 5 epochs. The default is 5.
        verbose : TYPE boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        Losses

        '''

        assert (len(xt.shape) > 1), 'Input data needs to be at least 2D'

        losses = []
        for i in range(iterations):

            if i % interval == 0 and i < cutoff:
                param_num = 0  # only update omegas that are note the first self.num_fourier_modes idxs of each param
                for num_freqs in self.num_freqs:
                    for k in range(param_num + self.num_fourier_modes, num_freqs):
                        self.fft(xt, k, verbose=verbose)
                    param_num += num_freqs

            if verbose:
                print('Iteration ', i)
                print(2 * np.pi / self.omegas)

            l = self.sgd(xt, i, weight_decay=weight_decay, verbose=verbose, lr_theta=lr_theta, lr_omega=lr_omega)
            losses.append(l)
            if verbose:
                print('Loss: ', l)

        return losses

    def predict(self, T):
        '''
        Predicts the data from 1 to T.

        Parameters
        ----------
        T : TYPE int
            Prediction horizon

        Returns
        -------
        TYPE numpy.array
            xhat from 0 to T.

        '''

        t = torch.arange(T, device=self.device) + 1
        ts_ = torch.unsqueeze(t, -1).type(torch.get_default_dtype())

        o = torch.unsqueeze(self.omegas, 0)
        k = torch.cat([torch.cos(ts_ * o), torch.sin(ts_ * o)], -1)

        if self.multi_gpu:
            params = self.model_obj.module.decode(k)
        else:
            params = self.model_obj.decode(k)

        return tuple(param.cpu().detach().numpy() for param in params)


class ModelObject(nn.Module):
    
    def __init__(self, num_freqs):
        super(ModelObject, self).__init__()
        self.num_freqs = num_freqs
        self.total_freqs = sum(num_freqs)

    def forward(self, y, x):
        """
        Forward computes the error.
        
        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
                
            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        """

        raise NotImplementedError()
    
    def decode(self, y):
        """
        Evaluates f at temporal snapshots y
        
        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
                
            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        """
        raise NotImplementedError()


class SkewNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n):
        """
        neural network that takes a vector of sines and cosines and produces a skew-normal distribution with parameters
        mu, sigma, and alpha (the outputs of the NN)
        :param x_dim: number of dimensions spanned by the probability distr
        :param num_freqs: list. number of frequencies used for each of the 3 parameters: [num_mu, num_sig, num_alpha]
        :param n: size of NN's second layer
        """
        super(SkewNLL, self).__init__(num_freqs)

        self.l1_mu = nn.Linear(2 * self.num_freqs[0], n)
        self.l2_mu = nn.Linear(n, 64)
        self.l3_mu = nn.Linear(64, x_dim)

        self.l1_sig = nn.Linear(2 * self.num_freqs[1], n)
        self.l2_sig = nn.Linear(n, 64)
        self.l3_sig = nn.Linear(64, x_dim)

        self.l1_a = nn.Linear(2 * self.num_freqs[2], n)
        self.l2_a = nn.Linear(n, 64)
        self.l3_a = nn.Linear(64, x_dim)

    def decode(self, w):
        w_mu = w[..., :2 * self.num_freqs[0]]
        y1 = nn.Tanh()(self.l1_mu(w_mu))
        y2 = nn.Tanh()(self.l2_mu(y1))
        y = self.l3_mu(y2)

        w_sigma = w[..., 2 * self.num_freqs[0]:2 * self.num_freqs[0] + 2 * self.num_freqs[1]]
        z1 = nn.Tanh()(self.l1_sig(w_sigma))
        z2 = nn.Tanh()(self.l2_sig(z1))
        z = 1000 * nn.Softplus()(self.l3_sig(z2))  # std should start big to avoid infinite gradients

        w_a = w[..., -2 * self.num_freqs[2]:]
        a1 = nn.Tanh()(self.l1_a(w_a))
        a2 = nn.Tanh()(self.l2_a(y1))
        a = self.l3_a(y2)

        return y, z, a

    def forward(self, w, data):
        y, z, a = self.decode(w)
        norm = torch.distributions.normal.Normal(0, 1)
        return -torch.mean((-(data - y)**2 / (2 * z**2)) - z.log() + norm.cdf(a * (data - y) / abs(z)).log(), dim=-1)
