import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from Encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from Decoder import Decoder, DecoderLayer
from Attention import FullAttention, ProbAttention, AttentionLayer
from Embedding import DataEmbedding



class Informer(nn.Module):
    def __init__(self, enc_in=2, dec_in=1, c_out=1, seq_len=168, label_len=48, out_len=24,
                 factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=256,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #        print(x_enc.shape)
        #        print(x_mark_enc.shape)
        x_enc = x_enc.permute(0, 2, 3, 1)
        x_enc = x_enc.reshape((x_enc.shape[0], x_enc.shape[1] * x_enc.shape[2], x_enc.shape[3]))

        enc_out = self.enc_embedding(x_enc)  # 将特征与data做embedding（包括三部分）
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        #        print(x_dec.shape)
        #        print(x_mark_dec.shape)
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #        print(dec_out.shape)
        dec_out = self.projection(dec_out)
        #        print(dec_out.shape)
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


#torch.manual_seed(5)
#model = Informer()
#inputs_1 = torch.randn((2, 168, 2))
#inputs_2 = torch.cat((torch.ones((2, 24, 1)), torch.zeros(2, 48, 1)), dim=1)
#inputs_2 = torch.zeros((2, 24, 1))
#outputs = model(inputs_1, inputs_2)
#print(outputs)
