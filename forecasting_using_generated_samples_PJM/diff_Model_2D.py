import abc
import torch

from typing import Callable, Union, Tuple

from torch import Tensor
from torch.nn import Module
import numpy as np


# numda to make the conditional distribution more concentrated
numda = 10
torch.manual_seed(2)
np.random.seed(2)

class SDE(abc.ABC):
    def __init__(self, N: int):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self) -> int:
        """ At the termination time of the forward SDE, the direction of time flow throughout the entire process is 0 → T """
        pass

    @abc.abstractmethod
    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ Drift and Diffusion Coefficients: f, g """
        pass

    @abc.abstractmethod
    def p_0t(self, x: Tensor, t: Tensor):
        """ return mean and std of p(x(t) | x(0)) """
        pass

    def prior_sampling(self, shape) -> Tensor:
        """ sampling from the prior distribution p_T(x) """
        torch.manual_seed(2)
        np.random.seed(2)
        return torch.randn(*shape)

    def discretize(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ discretize and return $f * \Delta t$, $g * \sqrt{\Delta}t$ """

        delta_t = 1. / self.N
        f, g = self.sde(x, t)

        return f * delta_t, g * torch.tensor(delta_t).to(t).sqrt()

    def reverse(self, score_fn: Union[Module, Callable], type='hotwave'):
        """ reverse-time SDE """

        N = self.N
        T = self.T

        # Drift and Diffusion Coefficients of forward process
        fw_sde = self.sde
        fw_discretize = self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.type = type

            @property
            def T(self) -> int:
                return T

            def sde(self, x: Tensor, t: Tensor, discrete: bool = False) -> Tuple[Tensor, Tensor]:

                f, g = fw_discretize(x, t) if discrete else fw_sde(x, t)
                #x.requires_grad = True
                score, classification = score_fn(x, t)

                if self.type == 'None':
                    conditional_gradient = torch.zeros_like(x)
                elif self.type == 'coldwave':
                    #print(x.requires_grad)
                    classification = torch.nn.Softmax(dim=1)(classification)
                    conditional_pro = torch.sum(classification[:, 0])
                    conditional_pro.backward(retain_graph=True)
                    conditional_gradient = x.grad
                elif self.type == 'hotwave':
                    classification = torch.nn.Softmax(dim=1)(classification)
                    conditional_pro = torch.sum(classification[:, 1])
                    conditional_pro.backward(retain_graph=True)
                    conditional_gradient = x.grad
                else:
                    classification = torch.nn.Softmax(dim=1)(classification)
                    conditional_pro = torch.sum(classification[:, 2])
                    conditional_pro.backward(retain_graph=True)
                    conditional_gradient = x.grad

                # Drift and Diffusion Coefficients of reverse-time SDE
                f = f - g[:, None, None, None] ** 2 * (score+numda*conditional_gradient)

                return f, g

        return RSDE()


def sde_loss_fn(sde: SDE, score_fn: Union[Module, Callable],
                data: Tensor, label: Tensor,
                eps: float = 1e-5) -> Tensor:


    bs = data.size(0)

    T = sde.T
    # set an eps for stability

    t = torch.rand(bs, device=data.device) * (T - eps) + eps

    # sampling noise from the standard Guassian distribution
    noise = torch.randn_like(data)
    mean, std = sde.p_0t(data, t)
    # Generate data with added noise
    perturbed_data = mean + std[:, None, None, None] * noise

    # estimate the score
    score, classification = score_fn(perturbed_data, t)
    loss = ((score * std[:, None, None, None] + noise) ** 2).reshape(bs, -1).mean(dim=1)
    ETL = torch.nn.CrossEntropyLoss()(classification, label)

    return loss.mean()+ETL


def classifier_fn(sde: SDE, classifier: Module, data: Tensor, label: Tensor, eps: float = 1e-5) -> Tensor:


    bs = data.size(0)

    T = sde.T
    t = torch.rand(bs, device=data.device) * (T - eps) + eps


    noise = torch.randn_like(data)
    mean, std = sde.p_0t(data, t)
    perturbed_data = mean + std[:, None, None, None] * noise

    # predict the extreme heat or cold wave label
    score = classifier(perturbed_data)
    loss = torch.nn.CrossEntropyLoss()(score, label)

    return loss



class VESDE(SDE):
    def __init__(self, sigma_min: float = 0.01,sigma_max: float = 50., N: int = 1000):
        super().__init__(N)

        # minimal noise scale
        self.sigma_min = sigma_min
        # maximum noise scale
        self.sigma_max = sigma_max

        # noise scales
        self.N = N
        # a power sequence forms an arithmetic sequence, the final result is a geometric sequence
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(sigma_min), np.log(sigma_max), N)
        )

    @property
    def T(self) -> int:
        # t  [0,1]
        return 1

    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ Drift and Diffusion Coefficients of VE-SDE """

        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t

        f = torch.zeros_like(x)
        g = sigma_t * torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=x.device).sqrt()

        return f, g

    def p_0t(self, x_0, t) -> Tuple[Tensor, Union[float, Tensor]]:
        """  VE SDE: mean and std of perturbation kernel"""

        return x_0, self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def prior_samping(self, shape) -> Tensor:
        """ $\mathcal N(0, sigma_max^2 I)$ """
        torch.manual_seed(2)
        np.random.seed(2)
        return torch.randn(*shape) * self.sigma_max

    def discretize(self, x: Tensor, t: Tensor):


        timestep_i = (t / self.T * (self.N - 1)).long()
        sigma_i = self.discrete_sigmas.to(x.device)[timestep_i]
        # $\sigma_{i-1}$
        adj_sigma = torch.where(
            timestep_i == 0,
            torch.zeros_like(sigma_i),
            self.discrete_sigmas.to(sigma_i.device)[timestep_i - 1]
        )

        f = torch.zeros_like(x)
        g = (sigma_i ** 2 - adj_sigma ** 2).sqrt()

        return f, g


class VPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., N: int = 2000):
        super().__init__(N)

        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)

        self.alphas = 1 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_1m_alphas_cumprod = (1. - self.alphas_cumprod).sqrt()

    @property
    def T(self) -> int:
        return 1

    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ Drift and Diffusion Coefficients of VP-SDE """

        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        f = -0.5 * beta_t[:, None, None, None] * x
        g = beta_t.sqrt()

        return f, g

    def p_0t(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Union[float, Tensor]]:
        """ VP-SDE: mean and std of perturbation kernel"""

        exponential = -0.25 * t ** 2 * \
                      (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        mean = torch.exp(exponential[:, None, None, None]) * x_0
        std = (1. - torch.exp(2. * exponential)).sqrt()

        return mean, std

    def discretize(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:

        timestep_i = (t / self.T * (self.N - 1)).long()

        sqrt_beta = self.discrete_betas.to(x.device)[timestep_i].sqrt()
        sqrt_alpha = self.alphas.to(x.device)[timestep_i].sqrt()

        f = (sqrt_alpha - 1.)[:, None, None, None] * x
        g = sqrt_beta

        return f, g


class Predictor(abc.ABC):
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], type='hotwave'):
        super().__init__()

        self.sde = sde
        self.rsde = sde.reverse(score_fn, type)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        pass


class Corrector(abc.ABC):
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], snr: float, n_steps: int):
        super().__init__()

        self.sde = sde
        self.score_fn = score_fn

        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        pass


def pc_sampling(
        sde: SDE, sample_shape,
        predictor_fn: Callable, corrector_fn: Callable,
        eps: float = 1e-3, denoise: bool = True,
        device: Union[str, int] = "cuda", type='hotwave'
) -> Tensor:
    x = sde.prior_sampling(sample_shape).to(device)
    x.requires_grad = True
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

    torch.manual_seed(2)
    np.random.seed(2)

    for t in timesteps:
        #torch.cuda.empty_cache()
        t = t.repeat(x.size(0))
        x, x_mean = corrector_fn(x, t, type)
        x1 = torch.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        x1.data = x.data.clone()
        x1.requires_grad = True
        x = x1
        x, x_mean = predictor_fn(x, t)
        x1 = torch.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        x1.data = x.data.clone()
        x1.requires_grad = True
        x = x1

    return x_mean if denoise else x


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], type='hotwave'):
        super().__init__(sde, score_fn, type=type)

    def update_fn(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        f_delta_t, g_sqrt_delta_t = self.rsde.sde(x, t, discrete=True)

        x_mean = x - f_delta_t
        x = x_mean + g_sqrt_delta_t[:, None, None, None] * torch.randn_like(x)

        return x, x_mean



class LangevinDynamicsCorrector(Corrector):
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], snr: float, n_steps: int):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x: Tensor, t: Tensor, type='hotwave') -> Tuple[Tensor, Tensor]:
        if isinstance(self.sde, VPSDE):
            timestep = (t / self.sde.T * (self.sde.N - 1)).long()
            alpha = self.sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        def get_norm(ts: Tensor) -> Tensor:
            return torch.norm(ts.reshape(ts.size(0), -1), dim=-1).mean()

        for _ in range(self.n_steps):
            x.requires_grad = True
            score, classification = self.score_fn(x, t)

            if type == 'None':
                conditional_gradient = torch.zeros_like(x)
            elif type == 'coldwave':
                x.requires_grad = True
                classification = torch.nn.Softmax(dim=1)(classification)
                conditional_pro = torch.sum(classification[:, 0])
                conditional_pro.backward()
                conditional_gradient = x.grad
            elif type == 'hotwave':
                classification = torch.nn.Softmax(dim=1)(classification)
                conditional_pro = torch.sum(classification[:, 1])
                conditional_pro.backward(retain_graph=True)
                conditional_gradient = x.grad
            else:
                classification = torch.nn.Softmax(dim=1)(classification)
                conditional_pro = torch.sum(classification[:, 2])
                conditional_pro.backward(retain_graph=True)
                conditional_gradient = x.grad



            z = torch.randn_like(x)

            # (B,)
            score_norm = get_norm(score+numda*conditional_gradient)
            # (B,)
            z_norm = get_norm(z)
            # (B,)
            step_size = 2 * alpha * (self.snr * z_norm / score_norm) ** 2

            x_mean = x + step_size[:, None, None, None] * (score+numda*conditional_gradient)
            x = x_mean + torch.sqrt(2 * step_size)[:, None, None, None] * z

        return x, x_mean


class NoneCorrector(Corrector):
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], snr: float, n_steps: int):
        pass

    def update_fn(self, x: Tensor, t: Tensor, type='hotwave') -> Tuple[Tensor, Tensor]:

        return x, x
