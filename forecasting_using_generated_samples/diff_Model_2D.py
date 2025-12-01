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
        """ 正向 SDE 的终止时刻, 整个过程时间的流动方向是 0 -> T """
        pass

    @abc.abstractmethod
    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ 计算漂移和扩散系数: f, g """
        pass

    @abc.abstractmethod
    def p_0t(self, x: Tensor, t: Tensor):
        """ 计算决定条件分布 p(x(t) | x(0)) 的参数，这里计划会返回均值和标准差 """
        pass

    def prior_sampling(self, shape) -> Tensor:
        """ 从先验分布 p_T(x) 中采样(作为采样起点)，先验通常为标准高斯分布 """
        #torch.manual_seed(2)
        #np.random.seed(2)
        return torch.randn(*shape)

    def discretize(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ 使用数值方法对 SDE 实施离散化, 返回 $f * \Delta t$, $g * \sqrt{\Delta}t$ """

        # 欧拉-丸山法所对应的离散时间间隔
        delta_t = 1. / self.N
        f, g = self.sde(x, t)

        return f * delta_t, g * torch.tensor(delta_t).to(t).sqrt()

    def reverse(self, score_fn: Union[Module, Callable], type='coldwave'):
        """ 构造逆向时间的 SDE/ODE, 返回1个代表 reverse-time SDE 的对象 """

        N = self.N
        T = self.T

        # 用于计算正向 SDE 的漂移和扩散系数的函数
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
                # 正向 SDE 的漂移和扩散系数
                f, g = fw_discretize(x, t) if discrete else fw_sde(x, t)
                #x.requires_grad = True
                score, classification = score_fn(x, t)

                if x.grad is not None:
                    x.grad.zero_()

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

                # 根据 reverse-time SDE 公式重新计算漂移系数
                f = f - g[:, None, None, None] ** 2 * (score+numda*conditional_gradient)

                return f, g

        return RSDE()


def sde_loss_fn(sde: SDE, score_fn: Union[Module, Callable],
                data: Tensor, label: Tensor,
                eps: float = 1e-5) -> Tensor:
    """ loss 函数, 其中时间变量是连续数值而非离散的时间步 """

    bs = data.size(0)

    T = sde.T
    # 时间变量从连续的均匀分布中采样
    # 这里做了特殊处理，使得最小值为 eps 而非 0,
    # 有助于稳定训练效果
    t = torch.rand(bs, device=data.device) * (T - eps) + eps

    # 从标准高斯分布中采样噪声
    noise = torch.randn_like(data)
    mean, std = sde.p_0t(data, t)
    # 生成加噪后的数据, 其服从均值为 mean, 标准差为 std 的高斯分布
    perturbed_data = mean + std[:, None, None, None] * noise

    # 模型根据含噪声的数据及当前时间估计出对应的 score
    score, classification = score_fn(perturbed_data, t)
    # loss 函数化简后的形式, 计算出 loss 后独立在每个样本的所有维度求平均
    loss = ((score * std[:, None, None, None] + noise) ** 2).reshape(bs, -1).mean(dim=1)
    ETL = torch.nn.CrossEntropyLoss()(classification, label)
    # 最后返回所有样本的 loss 均值
    return loss.mean()+ETL


def classifier_fn(sde: SDE, classifier: Module, data: Tensor, label: Tensor, eps: float = 1e-5) -> Tensor:
    """ loss 函数, 其中时间变量是连续数值而非离散的时间步 """

    bs = data.size(0)

    T = sde.T
    # 时间变量从连续的均匀分布中采样
    # 这里做了特殊处理，使得最小值为 eps 而非 0,
    # 有助于稳定训练效果
    t = torch.rand(bs, device=data.device) * (T - eps) + eps

    # 从标准高斯分布中采样噪声
    noise = torch.randn_like(data)
    mean, std = sde.p_0t(data, t)
    # 生成加噪后的数据, 其服从均值为 mean, 标准差为 std 的高斯分布
    perturbed_data = mean + std[:, None, None, None] * noise

    # 模型根据含噪声的数据及当前时间估计出对应的 score
    score = classifier(perturbed_data)
    #print(score)
    #score = torch.nn.Softmax()(score)
    # loss 函数化简后的形式, 计算出 loss 后独立在每个样本的所有维度求平均
    loss = torch.nn.CrossEntropyLoss()(score, label)
    #print(label)
    #print(loss)
    # 最后返回所有样本的 loss 均值
    return loss



class VESDE(SDE):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50., N: int = 1000):
        super().__init__(N)

        # 最小的 noise scale
        self.sigma_min = sigma_min
        # 最大的 noise scale
        self.sigma_max = sigma_max

        # NCSN 的 N 个 noise scales
        self.N = N
        # 幂形成等差数列, 则最终结果就是等比数列
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(sigma_min), np.log(sigma_max), N)
        )

    @property
    def T(self) -> int:
        # 在由离散的 SMLD 拓展至连续的 VE SDE 后，
        # 时间的 t 的取值范围为 [0,1]
        return 1

    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ 计算 VE SDE 的漂移和扩散系数, 对应论文公式(30) """

        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t

        f = torch.zeros_like(x)
        g = sigma_t * torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=x.device).sqrt()

        return f, g

    def p_0t(self, x_0, t) -> Tuple[Tensor, Union[float, Tensor]]:
        """ 计算 VE SDE 的 perturbation kernel 均值和标准差, 参照论文公式(31) """

        return x_0, self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def prior_samping(self, shape) -> Tensor:
        """ 先验分布是 $\mathcal N(0, sigma_max^2 I)$ """
        torch.manual_seed(2)
        np.random.seed(2)
        return torch.randn(*shape) * self.sigma_max

    def discretize(self, x: Tensor, t: Tensor):
        """ VE SDE 的数值离散形式, 即 SMLD 加噪的马尔科夫链, 对应论文公式(8)
        相当于 sde() 方法的离散版本 """

        # 将当前连续的时间变量转换为离散的时间步
        timestep_i = (t / self.T * (self.N - 1)).long()
        sigma_i = self.discrete_sigmas.to(x.device)[timestep_i]
        # $\sigma_{i-1}$
        adj_sigma = torch.where(
            timestep_i == 0,
            torch.zeros_like(sigma_i),
            self.discrete_sigmas.to(sigma_i.device)[timestep_i - 1]
        )

        # 因为将 SMLD 的马尔科夫链看作是 VE SDE 的数值离散化过程，
        # 所以这里依照伊藤 SDE 的惯例返回漂移和扩散系数
        f = torch.zeros_like(x)
        g = (sigma_i ** 2 - adj_sigma ** 2).sqrt()

        return f, g


class VPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., N: int = 2000):
        super().__init__(N)

        # $\bar{\beta}$ VP SDE 的 \beta(0) 和 \beta(1)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

        # DDPM 的 N 个 noise scales ${\beta_i}$
        # 与 VE SDE 的 \bar{\beta} 的关系是:
        # $N \beta = \bar{\beta}$
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)

        # DDPM 加噪过程中使用的 \alpha_i
        self.alphas = 1 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_1m_alphas_cumprod = (1. - self.alphas_cumprod).sqrt()

    @property
    def T(self) -> int:
        return 1

    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ 计算 VP SDE 的漂移和扩散系数, 对应论文公式(32) """

        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        f = -0.5 * beta_t[:, None, None, None] * x
        g = beta_t.sqrt()

        return f, g

    def p_0t(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Union[float, Tensor]]:
        """ 计算 VP SDE 的 perturbation kernel 的均值和标准差, 参考论文公式(33) """

        exponential = -0.25 * t ** 2 * \
                      (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        mean = torch.exp(exponential[:, None, None, None]) * x_0
        std = (1. - torch.exp(2. * exponential)).sqrt()

        return mean, std

    def discretize(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ VP SDE 的离散过程, 即 DDPM 加噪的马尔科夫链过程, 对应论文公式(10) """

        timestep_i = (t / self.T * (self.N - 1)).long()

        sqrt_beta = self.discrete_betas.to(x.device)[timestep_i].sqrt()
        sqrt_alpha = self.alphas.to(x.device)[timestep_i].sqrt()

        # 因为将 DDPM 的马尔科夫链看作是 VP SDE 的数值离散化过程，
        # 所以这里依照伊藤 SDE 的惯例返回漂移和扩散系数
        f = (sqrt_alpha - 1.)[:, None, None, None] * x
        g = sqrt_beta

        return f, g


class Predictor(abc.ABC):
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], type='coldwave'):
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
        device: Union[str, int] = "cuda", type='coldwave'
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
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], type='coldwave'):
        super().__init__(sde, score_fn, type=type)

    def update_fn(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        f_delta_t, g_sqrt_delta_t = self.rsde.sde(x, t, discrete=True)

        x_mean = x - f_delta_t
        x = x_mean + g_sqrt_delta_t[:, None, None, None] * torch.randn_like(x)

        return x, x_mean



class LangevinDynamicsCorrector(Corrector):
    def __init__(self, sde: SDE, score_fn: Union[Module, Callable], snr: float, n_steps: int):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x: Tensor, t: Tensor, type='coldwave') -> Tuple[Tensor, Tensor]:
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

    def update_fn(self, x: Tensor, t: Tensor, type='coldwave') -> Tuple[Tensor, Tensor]:
        return x, x