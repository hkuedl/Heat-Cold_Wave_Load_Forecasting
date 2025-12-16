import abc
import torch

from typing import Callable, Union, Tuple

from torch import Tensor
import torch.nn as nn
from torch.nn import Module
import numpy as np
from torch.autograd import Variable
torch.manual_seed(2)
np.random.seed(2)

device = "cuda:2"

class source_encoder(Module):
    def __init__(self, input_dim=2, hidden_size=64):
        super().__init__()

        self.fc1 = nn.Linear(input_dim * 168, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.exfc = nn.Linear(24, hidden_size)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.act = nn.ReLU()

    def forward(self, x, ex):
        h = self.act(self.fc1(x.view(x.shape[0], -1)))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        # h = self.act(h)

        hex = self.exfc(ex)
        #hex = self.act(hex)
        h = torch.cat((h, hex), dim=1)
        #h = self.act(h)
        #h = self.fc4(h)
        #h = self.act(h)

        return h

class source_encoder_LSTM(Module):
    def __init__(self, input_dim=2, output_dim=24):
        super().__init__()

        self.lstm = nn.LSTM(batch_first=True, input_size=input_dim, hidden_size=64, num_layers=1)
        self.fc1 = nn.Linear(64, 64)
        self.act = nn.ReLU()
        self.exfc1 = nn.Linear(24, 64)

    def forward(self, x, ex):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        hex = self.exfc1(ex)
        # hex = self.act(hex)
        # h = self.act(self.fc1(h))
        h = self.fc1(h)
        h = torch.cat((h, hex), dim=1)

        return h

class source_encoder_CNN(Module):
    def __init__(self, in_channel: int = 2, channels=[32, 64], output_dim=24):
        super().__init__()

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=True, padding=1)
        self.gnorm1 = nn.GroupNorm(32, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=True, padding=1)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.maxpool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(channels[1] * 2 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.exfc = nn.Linear(24, 64)

        # The swish activation function
        self.act = nn.Sigmoid()

    def forward(self, x, ex):
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

        hex = self.exfc(ex)
        #hex = nn.ReLU()(hex)

        h2 = torch.cat((h2, hex), dim=1)

        # print('h2 shape: ', h2.shape)

        return h2

class source_encoder_convlstm(Module):
    def __init__(self, input_channels=2, hidden_channels=[32, 64], kernel_size=3, step=7,
                 effective_step=[j for j in range(7)]):
        super(source_encoder_convlstm, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.fc1 = nn.Linear(hidden_channels[-1] * 7 * 24, 64)
        self.exfc = nn.Linear(24, 64)
        self.fc2 = nn.Linear(128, 24)
        for i in range(self.num_layers):  # 定义一个多层的convLSTM（即多个convLSTMCell），并存放在_all_layers列表中
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, x_ex):

        internal_state = []
        outputs = []
        for step in range(self.step):  # 在每一个时步进行前向运算
            x = input
            for i in range(self.num_layers):  # 对多层convLSTM中的每一层convLSTMCell，依次进行前向运算
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:  # 如果是在第一个时步，则需要调用init_hidden进行convLSTMCell的初始化
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)  # 调用convLSTMCell的forward进行前向运算
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        # print(len(outputs))
        h1 = outputs[-1]
        h1 = nn.ReLU()(self.fc1(h1.view(h1.shape[0], -1)))
        hex = nn.ReLU()(self.exfc(x_ex))
        h2 = torch.cat((h1, hex), dim=1)

        return h2



class target_encoder(Module):
    def __init__(self, input_dim=2, hidden_size=64):
        super().__init__()

        self.fc1 = nn.Linear(input_dim * 168, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.exfc = nn.Linear(24, hidden_size)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.act = nn.ReLU()

    def forward(self, x, ex):
        h = self.act(self.fc1(x.view(x.shape[0], -1)))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        # h = self.act(h)

        hex = self.exfc(ex)
        #hex = self.act(hex)
        h = torch.cat((h, hex), dim=1)
        #h = self.act(h)
        #h = self.fc4(h)
        #h = self.act(h)

        return h

class target_encoder_LSTM(Module):
    def __init__(self, input_dim=2, output_dim=24):
        super().__init__()

        self.lstm = nn.LSTM(batch_first=True, input_size=input_dim, hidden_size=64, num_layers=1)
        self.fc1 = nn.Linear(64, 64)
        self.act = nn.ReLU()
        self.exfc1 = nn.Linear(24, 64)

    def forward(self, x, ex):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        hex = self.exfc1(ex)
        # hex = self.act(hex)
        # h = self.act(self.fc1(h))
        h = self.fc1(h)
        h = torch.cat((h, hex), dim=1)

        return h

class target_encoder_CNN(Module):
    def __init__(self, in_channel: int = 2, channels=[32, 64], output_dim=24):
        super().__init__()

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=False, padding=1)
        self.gnorm1 = nn.GroupNorm(32, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False, padding=1)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.maxpool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(channels[1] * 2 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.exfc = nn.Linear(24, 64)

        # The swish activation function
        self.act = nn.Sigmoid()

    def forward(self, x, ex):
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

        hex = self.exfc(ex)
        #hex = nn.ReLU()(hex)

        h2 = torch.cat((h2, hex), dim=1)

        # print('h2 shape: ', h2.shape)

        return h2






class shared_encoder(Module):
    def __init__(self, input_dim=2, hidden_size=128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim * 168, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.exfc = nn.Linear(24, hidden_size)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.act = nn.ReLU()

    def forward(self, x, ex):
        h = self.act(self.fc1(x.view(x.shape[0], -1)))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        # h = self.act(h)

        hex = self.exfc(ex)
        #hex = self.act(hex)
        h = torch.cat((h, hex), dim=1)
        #h = self.act(h)
        #h = self.fc4(h)
        #h = self.act(h)

        return h

class shared_encoder_LSTM(Module):
    def __init__(self, input_dim=2, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(batch_first=True, input_size=input_dim, hidden_size=hidden_size, num_layers=1)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.exfc1 = nn.Linear(24, hidden_size)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size*2)

    def forward(self, x, ex):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        hex = self.exfc1(ex)
        #hex = self.act(hex)
        #h = self.act(self.fc1(h))
        h = self.fc1(h)
        h = torch.cat((h, hex), dim=1)
        #h = self.act(h)
        #h = self.fc4(h)
        #h = self.act(h)

        return h

class shared_encoder_CNN(Module):
    def __init__(self, in_channel: int = 2, channels=[32, 64], hidden_size=64):
        super().__init__()

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=False, padding=1)
        self.gnorm1 = nn.GroupNorm(32, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False, padding=1)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.maxpool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(channels[1] * 2 * 8, 128)
        self.fc2 = nn.Linear(128, hidden_size)
        self.exfc = nn.Linear(24, hidden_size)

        # The swish activation function
        self.act = nn.Sigmoid()

    def forward(self, x, ex):
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

        hex = self.exfc(ex)
        #hex = nn.ReLU()(hex)

        h2 = torch.cat((h2, hex), dim=1)

        # print('h2 shape: ', h2.shape)

        return h2

class shared_encoder_convlstm(Module):
    def __init__(self, input_channels=2, hidden_channels=[32, 64], kernel_size=3, step=7,
                 effective_step=[j for j in range(7)]):
        super(shared_encoder_convlstm, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.fc1 = nn.Linear(hidden_channels[-1] * 7 * 24, 64)
        self.exfc = nn.Linear(24, 64)
        self.fc2 = nn.Linear(128, 24)
        for i in range(self.num_layers):  # 定义一个多层的convLSTM（即多个convLSTMCell），并存放在_all_layers列表中
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, x_ex):

        internal_state = []
        outputs = []
        for step in range(self.step):  # 在每一个时步进行前向运算
            x = input
            for i in range(self.num_layers):  # 对多层convLSTM中的每一层convLSTMCell，依次进行前向运算
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:  # 如果是在第一个时步，则需要调用init_hidden进行convLSTMCell的初始化
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)  # 调用convLSTMCell的forward进行前向运算
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        # print(len(outputs))
        h1 = outputs[-1]
        h1 = nn.ReLU()(self.fc1(h1.view(h1.shape[0], -1)))
        hex = nn.ReLU()(self.exfc(x_ex))
        h2 = torch.cat((h1, hex), dim=1)

        return h2




class shared_decoder(Module):
    def __init__(self, input_size=128):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.loadfc = nn.Linear(128, 24*7)
        self.temfc = nn.Linear(128, 24*7)
        self.exfc = nn.Linear(128, 24)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))

        load = self.loadfc(h)
        tem = self.temfc(h)
        ex = self.exfc(h)

        return load, tem, ex

class shared_decoder_1(Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.loadfc = nn.Linear(256, 24*7)
        self.temfc = nn.Linear(256, 24*7)
        self.exfc = nn.Linear(256, 24)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))

        load = self.loadfc(h)
        tem = self.temfc(h)
        ex = self.exfc(h)

        return load, tem, ex

class predictor(Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 24)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.fc3(h)

        return h

class predictor_1(Module):
    def __init__(self, input_size=256):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 24)

        self.act = nn.ReLU()

    def forward(self, x, x_p):
        x = torch.cat((x, x_p), dim=1)

        h = self.act(self.fc1(x))

        #h = self.act(self.fc2(h))
        h = self.fc3(h)

        return h




class Separation_network(Module):
    def __init__(self, input_dim=2, hidden_size=64):
        super().__init__()

        self.source_encoder = source_encoder(input_dim, hidden_size)
        self.target_encoder = target_encoder(input_dim, hidden_size)
        self.shared_encoder = shared_encoder(input_dim, hidden_size)
        self.shared_decoder = shared_decoder(hidden_size*2)
        #self.shared_decoder = shared_decoder_1()
        #self.source_predictor = predictor()
        #self.target_predictor = predictor()
        self.source_predictor = predictor_1(input_size=hidden_size*4)
        self.target_predictor = predictor_1(input_size=hidden_size*4)

    def forward(self, xs, xs_ex, xt, xt_ex):
        hpt = self.target_encoder(xt, xt_ex)
        hps = self.source_encoder(xs, xs_ex)

        hct = self.shared_encoder(xt, xt_ex)
        hcs = self.shared_encoder(xs, xs_ex)

        #load_t, tem_t, ex_t = self.shared_decoder(torch.cat((hct, hpt), dim=1))
        #load_s, tem_s, ex_s = self.shared_decoder(torch.cat((hcs, hps), dim=1))
        load_t, tem_t, ex_t = self.shared_decoder(hct+hpt)
        load_s, tem_s, ex_s = self.shared_decoder(hcs+hps)

        #prediction_s = self.source_predictor(hcs)
        #prediction_t = self.target_predictor(hct)
        prediction_s = self.source_predictor(hcs, hps)
        prediction_t = self.target_predictor(hct, hpt)

        return hpt, hps, hct, hcs, load_t, tem_t, \
            ex_t, load_s, tem_s, ex_s, \
            prediction_s, prediction_t

class Separation_network_LSTM(Module):
    def __init__(self, input_dim=2, hidden_size=64):
        super().__init__()

        self.source_encoder = source_encoder(input_dim, hidden_size)
        self.target_encoder = target_encoder(input_dim, hidden_size)
        self.shared_encoder = shared_encoder_LSTM(input_dim, hidden_size)
        self.shared_decoder = shared_decoder(hidden_size*2)
        #self.source_predictor = predictor()
        #self.target_predictor = predictor()
        self.source_predictor = predictor_1(input_size=hidden_size*4)
        self.target_predictor = predictor_1(input_size=hidden_size*4)

    def forward(self, xs, xs_ex, xt, xt_ex):
        hpt = self.target_encoder(xt, xt_ex)
        hps = self.source_encoder(xs, xs_ex)

        hct = self.shared_encoder(xt, xt_ex)
        hcs = self.shared_encoder(xs, xs_ex)

        # load_t, tem_t, ex_t = self.shared_decoder(torch.cat((hct, hpt), dim=1))
        # load_s, tem_s, ex_s = self.shared_decoder(torch.cat((hcs, hps), dim=1))
        load_t, tem_t, ex_t = self.shared_decoder(hct + hpt)
        load_s, tem_s, ex_s = self.shared_decoder(hcs + hps)

        #prediction_s = self.source_predictor(hcs)
        #prediction_t = self.target_predictor(hct)
        prediction_s = self.source_predictor(hcs, hps)
        prediction_t = self.target_predictor(hct, hpt)

        return hpt, hps, hct, hcs, load_t, tem_t, \
            ex_t, load_s, tem_s, ex_s, \
            prediction_s, prediction_t

class Separation_network_conv(Module):
    def __init__(self, input_dim):
        super().__init__()

        self.source_encoder = source_encoder(input_dim)
        self.target_encoder = target_encoder(input_dim)
        self.shared_encoder = shared_encoder_LSTM(input_dim)
        self.shared_decoder = shared_decoder()
        #self.source_predictor = predictor()
        #self.target_predictor = predictor()
        self.source_predictor = predictor_1()
        self.target_predictor = predictor_1()

    def forward(self, xs, xs_ex, xt, xt_ex):
        hpt = self.target_encoder(xt, xt_ex)
        hps = self.source_encoder(xs, xs_ex)

        hct = self.shared_encoder(xt, xt_ex)
        hcs = self.shared_encoder(xs, xs_ex)

        # load_t, tem_t, ex_t = self.shared_decoder(torch.cat((hct, hpt), dim=1))
        # load_s, tem_s, ex_s = self.shared_decoder(torch.cat((hcs, hps), dim=1))
        load_t, tem_t, ex_t = self.shared_decoder(hct + hpt)
        load_s, tem_s, ex_s = self.shared_decoder(hcs + hps)

        #prediction_s = self.source_predictor(hcs)
        #prediction_t = self.target_predictor(hct)
        prediction_s = self.source_predictor(hcs, hps)
        prediction_t = self.target_predictor(hct, hpt)

        return hpt, hps, hct, hcs, load_t, tem_t, \
            ex_t, load_s, tem_s, ex_s, \
            prediction_s, prediction_t

class Separation_network_CNN(Module):
    def __init__(self, input_dim=2, hidden_size=64):
        super().__init__()

        self.source_encoder = source_encoder(input_dim, hidden_size)
        self.target_encoder = target_encoder(input_dim, hidden_size)
        self.shared_encoder = shared_encoder_CNN(input_dim, hidden_size=hidden_size)
        self.shared_decoder = shared_decoder(hidden_size*2)
        #self.source_predictor = predictor()
        #self.target_predictor = predictor()
        self.source_predictor = predictor_1(input_size=hidden_size*4)
        self.target_predictor = predictor_1(input_size=hidden_size*4)

    def forward(self, xs, xs_ex, xt, xt_ex):
        hpt = self.target_encoder(xt, xt_ex)
        hps = self.source_encoder(xs, xs_ex)

        hct = self.shared_encoder(xt, xt_ex)
        hcs = self.shared_encoder(xs, xs_ex)

        # load_t, tem_t, ex_t = self.shared_decoder(torch.cat((hct, hpt), dim=1))
        # load_s, tem_s, ex_s = self.shared_decoder(torch.cat((hcs, hps), dim=1))
        load_t, tem_t, ex_t = self.shared_decoder(hct + hpt)
        load_s, tem_s, ex_s = self.shared_decoder(hcs + hps)

        #prediction_s = self.source_predictor(hcs)
        #prediction_t = self.target_predictor(hct)
        prediction_s = self.source_predictor(hcs, hps)
        prediction_t = self.target_predictor(hct, hpt)

        return hpt, hps, hct, hcs, load_t, tem_t, \
            ex_t, load_s, tem_s, ex_s, \
            prediction_s, prediction_t



def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差

		return: [	K_ss K_st
				K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    #bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    bandwidth_list = [float(i+1)**2 for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起

def mmd(source, target, kernel_mul=2, kernel_num=5, fix_sigma=5):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
    # 当不同的时候，就需要乘上上面的M矩阵
    return loss


def orthogonal_loss(source, target):
    A = torch.matmul(source, target.permute(1, 0))
    loss = torch.norm(A, p='fro')**2/(torch.norm(source, p='fro')*torch.norm(target, p='fro'))
    #loss = 0
    #for i in range(A.shape[0]):
     #   for j in range(A.shape[1]):
      #      loss += abs(A[i][j])
    return loss


class ANN_model(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(128, output_dim)
        self.act = lambda x: x * nn.ReLU()(x)
        self.exfc = nn.Linear(24, 64)

    def forward(self, x, ex):
        h = self.act(self.fc1(x.view(x.shape[0], -1)))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))

        hex = self.exfc(ex)
        h = torch.cat((h, hex), dim=1)

        h = self.fc4(h)

        return h


class LSTM_model(Module):
    def __init__(self, input_dim=2, output_dim=24):
        super().__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=input_dim, hidden_size=128, num_layers=1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(256, output_dim)
        self.act = nn.ReLU()
        self.exfc1 = nn.Linear(24, 128)

    def forward(self, x, ex):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        #hex = self.act(self.exfc1(ex))
        hex = self.exfc1(ex)
        #h = self.act(self.fc1(h))
        #h = self.fc1(h)
        h = torch.cat((h, hex), dim=1)
        h = self.act(h)
        h = self.fc2(h)

        return h


class CNN_Model(nn.Module):
    def __init__(self, in_channel: int = 2, channels=[32, 64], output_dim=24):
        super().__init__()

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=True, padding=1)
        self.gnorm1 = nn.GroupNorm(32, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=True, padding=1)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.maxpool = nn.MaxPool2d(3, 3)

        self.fc1 = nn.Linear(channels[1]*2*8, 128)
        #self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, output_dim)
        self.exfc = nn.Linear(24, 128)

        # The swish activation function
        self.act = nn.ReLU()

    def forward(self, x, ex):
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
        #h2 = self.fc2(h2)
        h2 = nn.ReLU()(h2)


        hex = self.exfc(ex)
        hex = nn.ReLU()(hex)

        h2 = torch.cat((h2, hex), dim=1)

        h2 = self.fc3(h2)

        #print('h2 shape: ', h2.shape)

        return h2



# NBEATS
class NBEATS(nn.Module):
    def __init__(self, block_num=2, stack_num=2, loopback_window=168, future_horizen=24):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(NBEATS, self).__init__()
        self.name = 'apps.fmml.NbeatsModel'
        self.stack_num = stack_num
        block_stacks = []
        for _ in range(self.stack_num):
            block_stacks.append(
                BlockStack(block_num, loopback_window, future_horizen)
            )
        self.block_stacks = nn.ModuleList(block_stacks)

    def forward(self, x, x_ex):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        x_hat_prev = x[:,:,0]
        y_sum = None
        x_hat = None
        for idx in self.block_stacks:
            y_hat, x_hat = idx(x_hat_prev)
            x_hat_prev = x_hat_prev-x_hat
            if y_sum is None:
                y_sum = y_hat
            else:
                y_sum = y_sum+y_hat
        return y_sum

class BasicBlock(nn.Module):
    def __init__(self, loopback_window=168, future_horizen=1):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(BasicBlock, self).__init__()
        self.loopback_window = loopback_window
        self.future_horizen = future_horizen
        theta_f_num = 5
        theta_b_num = 5
        # 组成块的全连接层
        fc_layers = [64, 64, 64, 64]
        self.h1 = nn.Linear(loopback_window, fc_layers[0])
        self.h2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.h3 = nn.Linear(fc_layers[1], fc_layers[2])
        self.h4 = nn.Linear(fc_layers[2], fc_layers[3])
        self.theta_f = nn.Linear(fc_layers[3], theta_f_num, bias=False)
        self.y_hat = nn.Linear(theta_f_num, future_horizen)
        #self.y_hat = nn.Linear(fc_layers[3], future_horizen, bias=False)
        self.theta_b = nn.Linear(fc_layers[3], theta_b_num, bias=False)
        self.x_hat = nn.Linear(theta_b_num, self.loopback_window)
        #self.x_hat = nn.Linear(fc_layers[3], self.loopback_window, bias=False)

    def forward(self, x):
        h1 = self.h1(x)
        h1 = nn.ReLU()(h1)
        h2 = self.h2(h1)
        h2 = nn.ReLU()(h2)
        h3 = self.h3(h2)
        h3 = nn.ReLU()(h3)
        h4 = self.h4(h3)
        h4 = nn.ReLU()(h4)
        theta_f = self.theta_f(h4)
        y_hat = self.y_hat(theta_f)
        theta_b = self.theta_b(h4)
        x_hat = self.x_hat(theta_b)
        return y_hat, x_hat

class BlockStack(nn.Module):
    def __init__(self, block_num=3, loopback_window=168, future_horizen=1):
        super(BlockStack, self).__init__()
        self.block_num = block_num
        basic_blocks = []
        for _ in range(self.block_num):
            basic_blocks.append(
                BasicBlock(loopback_window, future_horizen)
            )
        self.basic_blocks = nn.ModuleList(basic_blocks)

    def forward(self, x):
        x_hat_prev = x
        y_sum = None
        x_hat = None
        for idx in self.basic_blocks:
            y_hat, x_hat = idx(x_hat_prev)
            x_hat_prev = x_hat_prev - x_hat
            if y_sum is None:
                y_sum = y_hat
            else:
                y_sum = y_sum + y_hat
        return y_sum, x_hat


# Impactnet
class RCU_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(RCU_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.relu(x)
        x1 = x
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x + x1
        x = self.relu(x)

        return x

class impactnet(nn.Module):
    def __init__(self, feature_size=24, out_channels=64, kernel_size=5, stride=1):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(impactnet, self).__init__()
        self.gru1 = RCU_block(2, out_channels, kernel_size, stride=stride)
        self.gru2 = RCU_block(out_channels, out_channels, kernel_size, stride=stride)
        self.gru3 = RCU_block(out_channels, out_channels, kernel_size, stride=stride)
        self.gru4 = RCU_block(out_channels, out_channels, kernel_size, stride=stride)
        self.max_pool = nn.MaxPool1d(kernel_size=4)
        self.relu = nn.ReLU()
        self.convfc = nn.Linear(42*out_channels, 96)
        self.fc1 = nn.Linear(feature_size, 96)
        self.fc2 = nn.Linear(96, 96)
        self.outfc1 = nn.Linear(96*2, 64)
        self.outfc2 = nn.Linear(64, 24)

    def forward(self, x, x_n):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        x = self.gru1(x.permute(0, 2, 1))
        x = self.gru2(x)
        x = self.gru3(x)
        x = self.gru4(x)
        x = self.max_pool(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.convfc(x)

        x_n = self.fc1(x_n)
        x_n = nn.ReLU()(x_n)
        x_n = self.fc2(x_n)
        x_n = nn.ReLU()(x_n)

        h = torch.cat((x,x_n), dim=1)
        h = self.outfc1(h)
        h = nn.ReLU()(h)
        h = self.outfc2(h)

        return h




class ConvLSTM(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=[32, 64], kernel_size=3, step=7, effective_step=[j for j in range(7)]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.fc1 = nn.Linear(hidden_channels[-1]*7*24, 128)
        self.exfc = nn.Linear(24, 128)
        self.fc2 = nn.Linear(256, 24)
        for i in range(self.num_layers):        # 定义一个多层的convLSTM（即多个convLSTMCell），并存放在_all_layers列表中
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, x_ex):

        internal_state = []
        outputs = []
        for step in range(self.step):  # 在每一个时步进行前向运算
            x = input
            for i in range(self.num_layers):  # 对多层convLSTM中的每一层convLSTMCell，依次进行前向运算
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:  # 如果是在第一个时步，则需要调用init_hidden进行convLSTMCell的初始化
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)  # 调用convLSTMCell的forward进行前向运算
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        #print(len(outputs))
        h1 = outputs[-1]
        h1 = nn.ReLU()(self.fc1(h1.view(h1.shape[0], -1)))
        hex = nn.ReLU()(self.exfc(x_ex))
        h2 = self.fc2(torch.cat((h1, hex), dim=1))

        return h2

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]).to(device)),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(device))




#model = ConvLSTM().to(device)
#inputs = torch.randn((1, 2, 7, 24)).to(device)
#inputs2 = torch.randn((1, 24)).to(device)
#outputs = model(inputs, inputs2)
#print(outputs.shape)






#model = NBEATS()
#inputs = torch.randn((2, 168, 2))
#outputs = model(inputs)
#print(outputs.shape)

#model = impactnet()
#inputs_1 = torch.randn((2, 168, 2))
#inputs_2 = torch.randn((2, 24))
#outputs = model(inputs_1, inputs_2)
#print(outputs.shape)