import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from autoformer_Encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from autoformer_Decoder import Decoder, DecoderLayer
from autoformer_Attention import FullAttention, ProbAttention, AttentionLayer
from autoformer_Embedding import DataEmbedding
from utils.series_decomposition import moving_avg, series_decomp
from AutoCorrelation import AutoCorrelationLayer, AutoCorrelation


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class Autoformer(nn.Module):
    def __init__(self, enc_in=2, dec_in=2, c_out=2, seq_len=168, label_len=48, out_len=24,
                 factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=256,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), moving_avg=25):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                           dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq,
                                           dropout)
        self.fc = nn.Linear(24*2, 24)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )



    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init

        x_enc = x_enc.permute(0, 2, 3, 1)
        x_enc = x_enc.reshape((x_enc.shape[0], x_enc.shape[1] * x_enc.shape[2], x_enc.shape[3]))

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        h2 = dec_out[:, -self.pred_len:, :]
        h2 = self.fc(h2.reshape(h2.shape[0], -1))

        return h2




#torch.manual_seed(5)
#model = Autoformer().to('cuda')
#inputs_1 = torch.randn((2, 168, 2))
#inputs_2 = torch.cat((torch.ones((2, 24, 1)), torch.zeros(2, 48, 1)), dim=1)
#inputs_2 = torch.zeros((2, 72, 2))
#outputs = model(inputs_1, inputs_2)
#print(outputs.shape)
