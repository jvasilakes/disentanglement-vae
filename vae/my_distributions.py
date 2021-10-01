import torch
from torch import autograd
from torch import distributions


class LogparamNormal(distributions.Normal):
    """
    Converts log-scale variance parameters output by a neural net
    into R^+ required by torch.distributions.Normal.
    """
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        self.var = torch.exp(logvar)
        super(LogparamNormal, self).__init__(self.mu, self.var)
        self.param_names = ["mu", "logvar"]

    @classmethod
    def from_params(cls, dense_params):
        """
        The neural net which estimates the parameters outputs them as a single
        tensor, which needs to be split into the corresponding parameters
        for the class.
        """
        mu, logvar = dense_params.chunk(2, dim=1)
        #logvar = torch.tanh(logvar) * 2.5
        return cls(mu, logvar)

    @staticmethod
    def kl_divergence(mu=None, logvar=None):
        """
        KL divergence between the given Normal distribution and N(0,1)
        """
        #if mu is None or logvar is None:
        #    mu = self.mu
        #    logvar = self.logvar
        kl = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)
                    - 1 - logvar)
        # sum over dimensions, average over examples
        kl = kl.mean(0).sum()
        return kl


class Beta(autograd.Function):
    """
    Beta function. Credit goes to rachtsingh
    https://github.com/rachtsingh/lgamma/blob/master/functions/beta.py
    """
    def forward(self, a, b):
        beta_ab = (torch.lgamma(a) + torch.lgamma(b)
                   - torch.lgamma(a + b)).exp()
        self.save_for_backward(a, b, beta_ab)
        return beta_ab

    def backward(self, grad_output):
        a, b, beta_ab = self.saved_tensors
        digamma_ab = torch.polygamma(0, a + b)
        return (grad_output * beta_ab * (torch.polygamma(0, a) - digamma_ab),
                grad_output * beta_ab * (torch.polygamma(0, b) - digamma_ab))


class LogparamBeta(distributions.Beta):
    """
    Converts log-scale alpha and beta parameters output by a neural net
    into R^+ required by torch.distributions.Beta.
    """
    def __init__(self, log_concentration1, log_concentration0):
        self.log_concentration1 = log_concentration1
        self.alpha = torch.exp(log_concentration1)
        self.log_concentration0 = log_concentration0
        self.beta = torch.exp(log_concentration0)
        super(LogparamBeta, self).__init__(
            self.alpha, self.beta)
        self.param_names = ["alpha", "beta"]

    @classmethod
    def from_params(cls, dense_params):
        """
        The neural net which estimates the parameters outputs them as a single
        tensor, which needs to be split into the corresponding parameters
        for the class.
        """
        log_concentration1, log_concentration0 = dense_params.chunk(2, dim=1)
        #log_concentration1 = torch.tanh(log_concentration1) * 2.5
        #log_concentration0 = torch.tanh(log_concentration0) * 2.5
        return cls(log_concentration1, log_concentration0)

    @staticmethod
    def kl_divergence(alpha=None, beta=None):
        """
        KL divergence between the given Beta distribution and Beta(1,1)
        """
        #if alpha is None or beta is None:
        #    a = self.alpha
        #    b = self.beta
        beta_fn_res = Beta().forward(alpha, beta)
        kl = -torch.log(beta_fn_res) + (alpha - 1) * torch.digamma(alpha) \
            + (beta - 1) * torch.digamma(beta) + (2 - alpha - beta) \
            * torch.digamma(alpha + beta)
        # average over examples
        return kl.mean()


lookup = {"normal": LogparamNormal,
          "gaussian": LogparamNormal,
          "beta": LogparamBeta}
