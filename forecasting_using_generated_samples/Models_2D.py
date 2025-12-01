import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union, Tuple

device = "cuda:2"
torch.manual_seed(2)
np.random.seed(2)

class GaussianFourierProjection(nn.Module):
    """ Gaussian random features for encoding time steps. """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        # \omega \sim \mathcal N(0, s^2 I), s = 30.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """ A fully connected layer that reshapes outputs to feature maps. """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]



class ClassifierModel(nn.Module):
    def __init__(self, in_channel: int = 2, channels=[32, 64]):
        super().__init__()

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=False, padding=1)
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False, padding=1)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.maxpool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(channels[1]*2*8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        h1 = self.conv1(x)
        ## Incorporate information from t
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h2 = self.maxpool(h2)
        h2 = self.fc1(h2.view(h2.shape[0], -1))
        h2 = nn.ReLU()(h2)
        h2 = self.fc2(h2)
        h2 = nn.ReLU()(h2)
        h2 = self.fc3(h2)

        #print('h2 shape: ', h2.shape)

        return h2



class ScoreModel(nn.Module):
    """ A time-dependent score-based model built upon U-Net architecture. """

    def __init__(self, p_0t: Callable, in_channel: int = 3, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
            p_0t: A function that takes time t and gives the standard
                  deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels: The number of channels for feature maps of each resolution.
            embed_dim: The dimensionality of Gaussian random feature embeddings.
        """

        super().__init__()

        # Gaussian random feature embedding layer for time
        self.classifier = ClassifierModel(in_channel, channels)


        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=False, padding=1)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False, padding=1)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=False, padding=1)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=1, bias=False, padding=1)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=1, bias=False, padding=1)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=1, bias=False,
                                         padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=1, bias=False,
                                         padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], in_channel, 3, stride=1, padding=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

        self.p_0t = p_0t

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        classification = self.classifier(x)

        embed = self.act(self.embed(t))

        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        #print('h1 shape: ', h1.shape)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        #print('h2 shape: ', h2.shape)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        #print('h3 shape: ', h3.shape)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        #print('h4 shape: ', h4.shape)

        # Decoding path with kip connection from the encoding path
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        #print('h shape: ', h.shape)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        #print('h shape: ', h.shape)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h, h1], dim=1))

        # 由于真实 score 的尺度(2-norm)在 1/std 水平,
        # 因此这里用 1/std 来 rescale 模型输出, 就会鼓励模型的输出具有单位尺度
        std = self.p_0t(x, t)[1]
        h = h / std[:, None, None, None]

        return h, classification







#Model = ClassifierModel()
#inputs = torch.randn((1, 2, 8, 24))
#y = Model(inputs)

#x = torch.randn((3, 2), requires_grad=True)
#y = torch.randn((2, 1), requires_grad=True)
#z = torch.matmul(x, y)
#z = torch.nn.Softmax(dim=0)(z)
#c = torch.sum(z[:, 0])
#c.backward(retain_graph=True)
#print(x.grad)
#z.backward()



