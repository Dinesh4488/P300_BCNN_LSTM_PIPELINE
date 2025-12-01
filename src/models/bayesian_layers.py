# src/models/bayesian_layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
BayesianLinear: simple Bayes-by-Backprop diagonal-Gaussian variational posterior
- stores variational parameters (mu, rho) in float32 (stable)
- sampling uses reparameterization: w = mu + sigma * eps (eps float32)
- during forward we cast sampled weight to the activation dtype (e.g. float16) if needed
- provides helper methods log_prior() and log_variational_posterior() (approximate)
"""

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_std=0.05,
                 prior_sigma1=0.1, prior_sigma2=0.0005, prior_pi=0.5):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features

        # store variational params in float32 for numeric stability
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.b_mu = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        self.b_rho = nn.Parameter(torch.empty(out_features, dtype=torch.float32))

        # init
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        # initialize rho such that softplus(rho) ≈ init_std
        self.w_rho.data.fill_(math.log(math.exp(init_std) - 1.0))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_mu, -bound, bound)
        self.b_rho.data.fill_(math.log(math.exp(init_std) - 1.0))

        # prior (scale-mixture of two zero-mean gaussians)
        self.prior_sigma1 = float(prior_sigma1)
        self.prior_sigma2 = float(prior_sigma2)
        self.prior_pi = float(prior_pi)

    def _sigma(self, rho):
        # Ensure positivity; operate in float32
        return F.softplus(rho)

    def sample_weight(self, device, dtype=torch.float32):
        """Sample weight and bias using reparam trick (in float32), then cast to dtype."""
        eps_w = torch.randn_like(self.w_mu, device=device, dtype=torch.float32)
        eps_b = torch.randn_like(self.b_mu, device=device, dtype=torch.float32)
        w_sigma = self._sigma(self.w_rho)
        b_sigma = self._sigma(self.b_rho)
        w = self.w_mu + w_sigma * eps_w
        b = self.b_mu + b_sigma * eps_b
        if dtype != torch.float32:
            w = w.to(dtype)
            b = b.to(dtype)
        return w, b

    def forward(self, x, sample=True):
        """
        x: input tensor (any dtype). We perform linear(x, weight, bias).
        If sample==True, sample weights (stochastic forward). If sample==False, use means.
        """
        device = x.device
        target_dtype = x.dtype

        if self.training or sample:
            # sample in float32 for stability, cast to target dtype for matmul
            w, b = self.sample_weight(device=device, dtype=target_dtype)
        else:
            # use mu (cast to target dtype)
            w = self.w_mu.to(device=device, dtype=target_dtype)
            b = self.b_mu.to(device=device, dtype=target_dtype)
        return F.linear(x, w, b)

    def log_prior(self):
        """Approximate log prior of weight means (evaluate mixture at mu)."""
        # operate in float32
        w = self.w_mu.view(-1)
        b = self.b_mu.view(-1)
        sigma1 = self.prior_sigma1
        sigma2 = self.prior_sigma2
        pi = self.prior_pi
        # Gaussian densities
        coeff1 = 1.0 / (math.sqrt(2.0 * math.pi) * sigma1)
        coeff2 = 1.0 / (math.sqrt(2.0 * math.pi) * sigma2)
        term_w = torch.log(pi * coeff1 * torch.exp(-0.5 * (w ** 2) / (sigma1 ** 2)) +
                           (1.0 - pi) * coeff2 * torch.exp(-0.5 * (w ** 2) / (sigma2 ** 2)) + 1e-12).sum()
        term_b = torch.log(pi * coeff1 * torch.exp(-0.5 * (b ** 2) / (sigma1 ** 2)) +
                           (1.0 - pi) * coeff2 * torch.exp(-0.5 * (b ** 2) / (sigma2 ** 2)) + 1e-12).sum()
        return term_w + term_b

    def log_variational_posterior(self):
        """Approximate log q(w|theta) using analytic entropy of Gaussian (evaluated at mu)."""
        w_sigma = self._sigma(self.w_rho)
        b_sigma = self._sigma(self.b_rho)
        # analytic log-likelihood of Gaussian evaluated at mu (i.e., constant term from entropy)
        lw = (-0.5 * math.log(2 * math.pi) - torch.log(w_sigma)).sum()
        lb = (-0.5 * math.log(2 * math.pi) - torch.log(b_sigma)).sum()
        return lw + lb
