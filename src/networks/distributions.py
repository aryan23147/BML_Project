import math
import torch
import torch.nn.functional as F


class VariationalPosterior(torch.nn.Module):
    def __init__(self, mu, rho, device):
        super(VariationalPosterior, self).__init__()
        self.mu = mu.to(device)
        self.rho = rho.to(device)
        self.device = device
        # gaussian distribution to sample epsilon from
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return F.softplus(self.rho)

    def sample(self):
        sigma = self.sigma
        epsilon = torch.randn_like(self.rho)
        # reparameterization trick for sampling from posterior
        posterior_sample = self.mu + sigma * epsilon
        return posterior_sample

    def log_prob(self, input):
        sigma = self.sigma
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.mu) ** 2) / (2 * sigma ** 2)).sum()


class Prior(torch.nn.Module):
    '''
    Scaled Gaussian Mixtures for Priors (Algorithm 1, Line 9)
    '''
    def __init__(self, args):
        super(Prior, self).__init__()
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = float(args.pi)
        self.device = args.device

        self.s1 = torch.tensor([math.exp(-1. * self.sig1)], dtype=torch.float32, device=self.device)
        self.s2 = torch.tensor([math.exp(-1. * self.sig2)], dtype=torch.float32, device=self.device)

        self.gaussian1 = torch.distributions.Normal(0, self.s1)
        self.gaussian2 = torch.distributions.Normal(0, self.s2)

    def log_prob(self, input):
        input = input.to(self.device)
        lp1 = self.gaussian1.log_prob(input)
        lp2 = self.gaussian2.log_prob(input)
        if self.pi <= 0.0:
            return lp2.sum()
        if self.pi >= 1.0:
            return lp1.sum()
        return torch.logaddexp(math.log(self.pi) + lp1, math.log(1.0 - self.pi) + lp2).sum()
