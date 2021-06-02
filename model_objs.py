#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Alex Mallen (atmallen@uw.edu)
Built on code from Henning Lange (helange@uw.edu)
"""

import torch
from torch import nn
import numpy as np
from scipy.special import factorial


class ModelObject(nn.Module):

    def __init__(self, num_freqs):
        super(ModelObject, self).__init__()
        self.num_freqs = num_freqs
        self.total_freqs = sum(num_freqs)

        self.param_idxs = []
        cumul = 0
        for num_freq in self.num_freqs:
            idxs = np.concatenate([cumul + np.arange(num_freq), self.total_freqs + cumul + np.arange(num_freq)])
            self.param_idxs.append(idxs)
            cumul += num_freq

    def forward(self, w, data, training_mask=None):
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

    def decode(self, w):
        """
        Evaluates f at temporal snapshots y

        Input:
            w: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
        """
        raise NotImplementedError()

    def mean(self, params):
        """returns the mean of a distribution with the given params"""
        return params[0]

    def std(self, params):
        """returns the standard deviation of a distribution with the given params"""
        return np.ones(params[0].shape)


class GEFComSkewNLL(ModelObject):
    def __init__(self, x_dim, num_freqs, n):
        """
        neural network that takes a vector of sines and cosines and produces a skew-normal distribution with parameters
        mu, sigma, and alpha (the outputs of the NN). trains using NLL.
        :param x_dim: number of dimensions spanned by the probability distr
        :param num_freqs: list. number of frequencies used for each of the 3 parameters: [num_mu, num_sig, num_alpha]
        :param n: size of NN's second layer
        """
        super(GEFComSkewNLL, self).__init__(num_freqs)

        self.l1_mu = nn.Linear(2 * self.num_freqs[0] + 1, n)
        self.l2_mu = nn.Linear(n, 64)
        self.l3_mu = nn.Linear(64, x_dim)

        self.l1_sig = nn.Linear(2 * self.num_freqs[1] + 1, n)
        self.l2_sig = nn.Linear(n, 64)
        self.l3_sig = nn.Linear(64, x_dim)

        self.l1_a = nn.Linear(2 * self.num_freqs[2] + 1, n)
        self.l2_a = nn.Linear(n, 32)
        self.l3_a = nn.Linear(32, x_dim)

        self.norm = torch.distributions.normal.Normal(0, 1)

    def decode(self, w):
        w_mu = w[..., (*self.param_idxs[0], -1)]
        y1 = nn.Tanh()(self.l1_mu(w_mu))
        y2 = nn.Tanh()(self.l2_mu(y1))
        y = self.l3_mu(y2)

        w_sigma = w[..., (*self.param_idxs[1], -1)]
        z1 = nn.Tanh()(self.l1_sig(w_sigma))
        z2 = nn.Tanh()(self.l2_sig(z1))
        z = 10 * nn.Softplus()(self.l3_sig(z2))  # start with large uncertainty to avoid small probabilities

        w_a = w[..., (*self.param_idxs[2], -1)]
        a1 = nn.Tanh()(self.l1_a(w_a))
        a2 = nn.Tanh()(self.l2_a(a1))
        a = self.l3_a(a2)

        return y, z, a

    def forward(self, w, data, training_mask=None):
        mu, sig, alpha = self.decode(w)
        if training_mask is None:
            y = mu
            z = sig
            a = alpha
        else:
            y = training_mask * mu + (1 - training_mask) * mu.detach()
            # z = (1 - training_mask) * sig + training_mask * sig.detach()
            # a = (1 - training_mask) * alpha + training_mask * alpha.detach()
            z = sig
            a = alpha

        losses = (data - y)**2 / (2 * z**2) + z.log() - self._norm_logcdf(a * (data - y) / z)
        avg = torch.mean(losses, dim=-1)
        return avg

    def _norm_logcdf(self, z):

        if (z < -7).any():  # these result in NaNs otherwise
            print("THIS BATCH USING LOG CDF APPROXIMATION (large z-score can otherwise cause numerical instability)")
            # https://stats.stackexchange.com/questions/106003/approximation-of-logarithm-of-standard-normal-cdf-for-x0/107548#107548?newreg=5e5f6365aa7046aba1c447e8ae263fec
            # I found this approx to be good: less than 0.04 error for all -20 < x < -5
            # approx = lambda x: -0.5 * x ** 2 - 4.8 + 2509 * (x - 13) / ((x - 40) ** 2 * (x - 5))
            ans = torch.where(z < -0.1, -0.5 * z ** 2 - 4.8 + 2509 * (z - 13) / ((z - 40) ** 2 * (z - 5)),
                                        -torch.exp(-z * 2) / 2 - torch.exp(-(z - 0.2) ** 2) * 0.2)
        else:
            ans = self.norm.cdf(z).log()

        return ans

    def mean(self, params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return mu + sigma * delta * (2 / np.pi) ** 0.5

    def std(self, params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return sigma * (1 - 2 * delta ** 2 / np.pi) ** 0.5


class SkewNLLwithTime(ModelObject):
    def __init__(self, x_dim, num_freqs, n=256):
        """
        neural network that takes a vector of sines and cosines and produces a skew-normal distribution with parameters
        mu, sigma, and alpha (the outputs of the NN). trains using NLL. Takes time as an input along with the vector of
        sines and cosines
        :param x_dim: number of dimensions spanned by the probability distr
        :param num_freqs: list. number of frequencies used for each of the 3 parameters: [num_mu, num_sig, num_alpha]
        :param n: size of NN's second layer
        """
        super(SkewNLLwithTime, self).__init__(num_freqs)

        self.l1_mu = nn.Linear(2 * self.num_freqs[0] + 1, n)
        self.l2_mu = nn.Linear(n, 64)
        self.l3_mu = nn.Linear(64, x_dim)

        self.l1_sig = nn.Linear(2 * self.num_freqs[1] + 1, n)
        self.l2_sig = nn.Linear(n, 64)
        self.l3_sig = nn.Linear(64, x_dim)

        self.l1_a = nn.Linear(2 * self.num_freqs[2] + 1, n)
        self.l2_a = nn.Linear(n, 32)
        self.l3_a = nn.Linear(32, x_dim)

        self.norm = torch.distributions.normal.Normal(0, 1)

    def decode(self, w):
        w_mu = w[..., (*self.param_idxs[0], -1)]
        y1 = nn.Tanh()(self.l1_mu(w_mu))
        y2 = nn.Tanh()(self.l2_mu(y1))
        y = self.l3_mu(y2)

        w_sigma = w[..., (*self.param_idxs[1], -1)]
        z1 = nn.Tanh()(self.l1_sig(w_sigma))
        z2 = nn.Tanh()(self.l2_sig(z1))
        z = 10 * nn.Softplus()(self.l3_sig(z2))  # start with large uncertainty to avoid small probabilities

        w_a = w[..., (*self.param_idxs[2], -1)]
        a1 = nn.Tanh()(self.l1_a(w_a))
        a2 = nn.Tanh()(self.l2_a(a1))
        a = self.l3_a(a2)

        return y, z, a

    def forward(self, w, data, training_mask=None):
        mu, sig, alpha = self.decode(w)
        if training_mask is None:
            y = mu
            z = sig
            a = alpha
        else:
            y = training_mask * mu + (1 - training_mask) * mu.detach()
            # z = (1 - training_mask) * sig + training_mask * sig.detach()
            # a = (1 - training_mask) * alpha + training_mask * alpha.detach()
            z = sig
            a = alpha

        losses = (data - y)**2 / (2 * z**2) + z.log() - self._norm_logcdf(a * (data - y) / z)
        avg = torch.mean(losses, dim=-1)
        return avg

    def _norm_logcdf(self, z):

        if (z < -7).any():  # these result in NaNs otherwise
            # print("THIS BATCH USING LOG CDF APPROXIMATION (large z-score can otherwise cause numerical instability)")
            # https://stats.stackexchange.com/questions/106003/approximation-of-logarithm-of-standard-normal-cdf-for-x0/107548#107548?newreg=5e5f6365aa7046aba1c447e8ae263fec
            # I found this approx to be good: less than 0.04 error for all -20 < x < -5
            # approx = lambda x: -0.5 * x ** 2 - 4.8 + 2509 * (x - 13) / ((x - 40) ** 2 * (x - 5))
            ans = torch.where(z < -0.1, -0.5 * z ** 2 - 4.8 + 2509 * (z - 13) / ((z - 40) ** 2 * (z - 5)),
                                        -torch.exp(-z * 2) / 2 - torch.exp(-(z - 0.2) ** 2) * 0.2)
        else:
            ans = self.norm.cdf(z).log()

        return ans

    def mean(self, params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return mu + sigma * delta * (2 / np.pi) ** 0.5

    def std(self, params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return sigma * (1 - 2 * delta ** 2 / np.pi) ** 0.5


class SkewNormalNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n):
        """
        neural network that takes a vector of sines and cosines and produces a skew-normal distribution with parameters
        mu, sigma, and alpha (the outputs of the NN). trains using NLL.
        :param x_dim: number of dimensions spanned by the probability distr
        :param num_freqs: list. number of frequencies used for each of the 3 parameters: [num_mu, num_sig, num_alpha]
        :param n: size of NN's second layer
        """
        super(SkewNormalNLL, self).__init__(num_freqs)

        self.l1_mu = nn.Linear(2 * self.num_freqs[0], n)
        self.l2_mu = nn.Linear(n, 64)
        self.l3_mu = nn.Linear(64, x_dim)

        self.l1_sig = nn.Linear(2 * self.num_freqs[1], n)
        self.l2_sig = nn.Linear(n, 64)
        self.l3_sig = nn.Linear(64, x_dim)

        self.l1_a = nn.Linear(2 * self.num_freqs[2], n)
        self.l2_a = nn.Linear(n, 32)
        self.l3_a = nn.Linear(32, x_dim)

        self.norm = torch.distributions.normal.Normal(0, 1)

    def decode(self, w):
        w_mu = w[..., self.param_idxs[0]]
        y1 = nn.Tanh()(self.l1_mu(w_mu))
        y2 = nn.Tanh()(self.l2_mu(y1))
        y = self.l3_mu(y2)

        w_sigma = w[..., self.param_idxs[1]]
        z1 = nn.Tanh()(self.l1_sig(w_sigma))
        z2 = nn.Tanh()(self.l2_sig(z1))
        z = 10 * nn.Softplus()(self.l3_sig(z2))  # start with large uncertainty to avoid small probabilities

        w_a = w[..., self.param_idxs[2]]
        a1 = nn.Tanh()(self.l1_a(w_a))
        a2 = nn.Tanh()(self.l2_a(a1))
        a = self.l3_a(a2)

        return y, z, a

    def forward(self, w, data, training_mask=None):
        mu, sig, alpha = self.decode(w)
        if training_mask is None:
            y = mu
            z = sig
            a = alpha
        else:
            y = training_mask * mu + (1 - training_mask) * mu.detach()
            # z = (1 - training_mask) * sig + training_mask * sig.detach()
            # a = (1 - training_mask) * alpha + training_mask * alpha.detach()
            z = sig
            a = alpha

        losses = (data - y)**2 / (2 * z**2) + z.log() - self._norm_logcdf(a * (data - y) / z)
        avg = torch.mean(losses, dim=-1)
        return avg

    def _norm_logcdf(self, z):

        if (z < -7).any():  # these result in NaNs otherwise
            # print("THIS BATCH USING LOG CDF APPROXIMATION (large z-score can otherwise cause numerical instability)")
            # https://stats.stackexchange.com/questions/106003/approximation-of-logarithm-of-standard-normal-cdf-for-x0/107548#107548?newreg=5e5f6365aa7046aba1c447e8ae263fec
            # I found this approx to be good: less than 0.04 error for all -20 < x < -5
            # approx = lambda x: -0.5 * x ** 2 - 4.8 + 2509 * (x - 13) / ((x - 40) ** 2 * (x - 5))
            ans = torch.where(z < -0.1, -0.5 * z ** 2 - 4.8 + 2509 * (z - 13) / ((z - 40) ** 2 * (z - 5)),
                                        -torch.exp(-z * 2) / 2 - torch.exp(-(z - 0.2) ** 2) * 0.2)
        else:
            ans = self.norm.cdf(z).log()

        return ans

    def mean(self, params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return mu + sigma * delta * (2 / np.pi) ** 0.5

    def std(self, params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return sigma * (1 - 2 * delta ** 2 / np.pi) ** 0.5


class NormalNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n=128, n2=64):
        """
        Negative Log Likelihood neural network assuming Gaussian distribution of x at every point in time.
        Trains using NLL and trains mu and sigma separately to prevent
        overfitting
        :param x_dim: dimension of what will be modeled
        :param num_freqs: list of the number of frequencies used to model each parameter: [num_mu, num_sigma]
        :param n: size of 2nd layer of NN
        """
        super(NormalNLL, self).__init__(num_freqs)

        self.l1_mu = nn.Linear(2 * self.num_freqs[0], n)
        self.l2_mu = nn.Linear(n, n2)
        self.l3_mu = nn.Linear(n2, x_dim)

        self.l1_sig = nn.Linear(2 * self.num_freqs[1], n)
        self.l2_sig = nn.Linear(n, n2)
        self.l3_sig = nn.Linear(n2, x_dim)

    def decode(self, w):
        w_mu = w[..., self.param_idxs[0]]
        y1 = nn.Tanh()(self.l1_mu(w_mu))
        y2 = nn.Tanh()(self.l2_mu(y1))
        y = self.l3_mu(y2)

        w_sigma = w[..., self.param_idxs[1]]
        z1 = nn.Tanh()(self.l1_sig(w_sigma))
        z2 = nn.Tanh()(self.l2_sig(z1))
        z = 10 * nn.Softplus()(self.l3_sig(z2))  # start big to avoid infinite gradients

        return y, z

    def forward(self, w, data, training_mask=None):
        mu, sig = self.decode(w)
        if training_mask is None:
            y = mu
            z = sig
        else:
            y = training_mask * mu + (1 - training_mask) * mu.detach()
            # z = (1 - training_mask) * sig + training_mask * sig.detach()
            z = sig

        losses = (data - y) ** 2 / (2 * z ** 2) + torch.log(z)
        avg = torch.mean(losses, dim=-1)
        return avg

    def mean(self, params):
        return params[0]

    def std(self, params):
        return params[1]


class ConwayMaxwellPoissonNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n, terms=20):
        """
        Negative Log Likelihood neural network assuming Conway Maxwell Poisson distribution of x at every point in time.
        This is a generalization of the discrete Poisson distribution. Trains using NLL.
        :param x_dim: dimension of what will be modeled
        :param num_freqs: list of the number of frequencies used to model each parameter: [num_rate, num_a]
        :param n: size of 2nd layer of NN
        """
        super(ConwayMaxwellPoissonNLL, self).__init__(num_freqs)

        self.l1_rate = nn.Linear(2 * self.num_freqs[0], n)
        self.l2_rate = nn.Linear(n, 64)
        self.l3_rate = nn.Linear(64, x_dim)

        self.l1_v = nn.Linear(2 * self.num_freqs[1], n)
        self.l2_v = nn.Linear(n, 64)
        self.l3_v = nn.Linear(64, x_dim)

        self.terms = terms

    def decode(self, w):
        w_rate = w[..., self.param_idxs[0]]
        rate1 = nn.Tanh()(self.l1_rate(w_rate))
        rate2 = nn.Tanh()(self.l2_rate(rate1))
        rate = 1 / nn.Softplus()(self.l3_rate(rate2))  # helps convergence

        w_v = w[..., self.param_idxs[1]]
        v1 = nn.Tanh()(self.l1_v(w_v))
        v2 = nn.Tanh()(self.l2_v(v1))
        v = nn.Softplus()(self.l3_v(v2))

        return rate, v

    def forward(self, w, data, training_mask=None):
        assert (training_mask is None), "Training masks won't help when using a CMP distribution"
        rate, v = self.decode(w)

        losses = -self._logCMPpmf(data, rate, v)
        avg = torch.mean(losses, dim=-1)
        return avg

    def _Z(self, rate, v):
        j = torch.arange(self.terms)
        return torch.sum(rate**j / (factorial(j)**v))

    def _logCMPpmf(self, x, rate, v):
        return x * torch.log(rate) - v * x.apply_(self._log_factorial) - torch.log(self._Z(rate, v))

    def _log_factorial(self, x):
        # log(x(x-1)(x-2)...(2)(1)) = log(x) + log(x-1) + ... + log(2) + log(1)
        # the hard part is vectorizing it, therefore it only takes scalar inputs
        return torch.sum(torch.log(torch.arange(1, x + 1)))

    def mean(self, params, num_terms=100):
        rate = params[0]
        v = params[1]
        terms = np.array([x * rate**x / (factorial(x)**v * self._npZ(rate, v)) for x in range(num_terms)])
        return np.sum(terms)

    def std(self, params, num_terms=100):
        rate = params[0]
        v = params[1]
        terms = np.array([x**2 * rate ** x / (factorial(x) ** v * self._npZ(rate, v)) for x in range(num_terms)])
        var = np.sum(terms) - self.mean(params, num_terms=num_terms)**2
        return np.sqrt(var)

    def _npZ(self, rate, v):
        j = np.arange(100)
        return np.sum(rate ** j / (factorial(j) ** v))


class GammaNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n):
        """
        Negative Log Likelihood neural network assuming Gamma distribution of x at every point in time.
        Trains using NLL
        :param x_dim: dimension of what will be modeled
        :param num_freqs: list of the number of frequencies used to model each parameter: [num_rate, num_a]
        :param n: size of 2nd layer of NN
        """
        super(GammaNLL, self).__init__(num_freqs)

        self.l1_rate = nn.Linear(2 * self.num_freqs[0], n)
        self.l2_rate = nn.Linear(n, 64)
        self.l3_rate = nn.Linear(64, x_dim)

        self.l1_a = nn.Linear(2 * self.num_freqs[1], n)
        self.l2_a = nn.Linear(n, 64)
        self.l3_a = nn.Linear(64, x_dim)

    def decode(self, w):
        w_rate = w[..., self.param_idxs[0]]
        rate1 = nn.Tanh()(self.l1_rate(w_rate))
        rate2 = nn.Tanh()(self.l2_rate(rate1))
        rate = 1 / nn.Softplus()(self.l3_rate(rate2))  # helps convergence

        w_a = w[..., self.param_idxs[1]]
        a1 = nn.Tanh()(self.l1_a(w_a))
        a2 = nn.Tanh()(self.l2_a(a1))
        a = nn.Softplus()(self.l3_a(a2))

        return rate, a

    def forward(self, w, data, training_mask=None):
        assert (training_mask is None), "Training masks won't help when using a Gamma distribution"
        rate, a = self.decode(w)

        losses = -torch.distributions.gamma.Gamma(a, rate).log_prob(data)
        avg = torch.mean(losses, dim=-1)
        return avg

    def mean(self, params):
        return params[1] / params[0]

    def std(self, params):
        return np.sqrt(params[1] / params[0] ** 2)


class PoissonNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n):
        """
        Negative Log Likelihood neural network assuming Poisson distribution of x at every point in time.
        Trains using NLL
        :param x_dim: dimension of what will be modeled
        :param num_freqs: list of the number of frequencies used to model each parameter: [num_rate,]
        :param n: size of 2nd layer of NN
        """
        super(PoissonNLL, self).__init__(num_freqs)

        self.l1_rate = nn.Linear(2 * self.num_freqs[0], n)
        self.l2_rate = nn.Linear(n, 64)
        self.l3_rate = nn.Linear(64, x_dim)

    def decode(self, w):
        w_rate = w[..., self.param_idxs[0]]
        rate1 = nn.Tanh()(self.l1_rate(w_rate))
        rate2 = nn.Tanh()(self.l2_rate(rate1))
        rate = 1 / nn.Softplus()(self.l3_rate(rate2))  # helps convergence

        return rate,

    def forward(self, w, data, training_mask=None):
        assert (training_mask is None), "Poisson distributions don't support training masks"
        rate, = self.decode(w)

        losses = -torch.distributions.poisson.Poisson(rate).log_prob(data)
        avg = torch.mean(losses, dim=-1)
        return avg

    def mean(self, params):
        return params[0]

    def std(self, params):
        return np.sqrt(params[0])
