import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils import weight_norm, remove_weight_norm

from latent_diffusion.modules.nn import avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module, checkpoint

def exists(val):
    return val is not None

class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, hidden_size, dim_feedforward, context_dim, dropout, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of improved part
        self.dropout = Dropout(dropout)
        self.linear_in = Linear(d_model, d_model*4)
        self.linear_out = Linear(d_model*4, d_model)

        self.to_k = nn.Sequential(
            Linear(context_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(d_model, d_model, bias=False))
        self.to_v = nn.Sequential(
            Linear(context_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(d_model, d_model, bias=False))
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, context=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        #print(context.shape, src.shape, flush=True)
        src2 = self.self_attn(src, self.to_k(context), self.to_v(context))[0]
        #src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        
        src2 = self.linear_out(self.dropout(self.activation(self.linear_in(src))))
        src = src + self.dropout2(src2)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class SingleTransformer(nn.Module):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
    """
    def __init__(self, input_size, hidden_size, context_dim, dropout):
        super(SingleTransformer, self).__init__()
        self.transformer = TransformerEncoderLayer(d_model=input_size, nhead=8, hidden_size=hidden_size,
                                                   dim_feedforward=hidden_size*2, context_dim=context_dim, dropout=dropout)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            Linear(
                input_size,
                2 * input_size,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            weight_norm(conv_nd(1, input_size, input_size, 3, padding=1))
        )

    def forward(self, x, emb, context):
        return checkpoint(self._forward, (x, emb, context), self.parameters(), True)

    def forward(self, x, emb, context=None):
        # input shape: batch, seq, dim
        x = self.transformer(x.permute(1, 0, 2).contiguous(),#).permute(1, 0, 2).contiguous()
                             context.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()

        emb_out = self.emb_layers(emb).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        x = x * (1 + scale) + shift
        x = self.out_layers(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return x


# dual-path transformer
class DPT(nn.Module):
    """
    Deep dual-path transformer.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        num_layers: int, number of stacked Transformer layers. Default is 1.
        dropout: float, dropout ratio. Default is 0.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, context_dim=512, dropout=0):
        super(DPT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # dual-path transformer
        self.row_transformer = nn.ModuleList([])
        self.col_transformer = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_transformer.append(SingleTransformer(input_size, hidden_size,
                context_dim if i >= num_layers//4 and i < (num_layers*3)//4 else input_size, dropout))
            self.col_transformer.append(SingleTransformer(input_size, hidden_size,
                context_dim if i >= num_layers//4 and i < (num_layers*3)//4 else input_size, dropout))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, x, emb, context=None):
        # input shape: batch, N, dim1, dim2
        # apply transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        #input = input.to(device)
        batch_size, _, dim1, dim2 = x.shape
        output = x
        #print(x.shape, flush=True)
        for i in range(self.num_layers):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_transformer[i](row_input, emb.repeat_interleave(dim2, 0),# None)
                context.repeat_interleave(dim2, 0) if i >= self.num_layers//4 and i < (self.num_layers*3)//4 else row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_transformer[i](col_input, emb.repeat_interleave(dim1, 0),# None)
                context.repeat_interleave(dim1, 0) if i >= self.num_layers//4 and i < (self.num_layers*3)//4 else col_input)  # B*dim2, dim1, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output
            #output = F.layer_norm(output, output.shape[1:])

        output = self.output(output) # B, output_size, dim1, dim2

        return output


# base module for deep DPT
class DPTNet(nn.Module):
    def __init__(self,
            input_dim=256,
            feature_dim=1024,
            hidden_dim=512,
            layer=8,
            segment_size=64,
            context_dim=512,
            extra_film_condition_dim=None,
            dropout=0,
            predict_xstart=False,
            learn_sigma=False,
            end2end=False,
        ):
        super(DPTNet, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size
        self.predict_xstart = predict_xstart
        self.learn_sigma = learn_sigma
        self.extra_film_condition_dim = extra_film_condition_dim
        self.eps = 1e-8

        self.time_embed = nn.Sequential(
            Linear(feature_dim, feature_dim),
            nn.SiLU(),
            Linear(feature_dim, feature_dim),
        )

        if extra_film_condition_dim is not None:
            self.film_emb = nn.Linear(self.extra_film_condition_dim, feature_dim)
            print(
                "+ Use extra condition on UNet channel using Film. Extra condition dimension is %s. "
                % self.extra_film_condition_dim
            )
        else:
            self.film_emb = None
        #self.context_embed = nn.Sequential(
        #    Linear(context_dim, feature_dim),
        #    nn.SiLU(),
        #    Linear(feature_dim, feature_dim),
        #)
        
        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)
        self.BN_gn = nn.GroupNorm(1, self.feature_dim, eps=1e-8)

        # DPT model
        self.DPT = DPT(self.feature_dim, self.hidden_dim, self.feature_dim,
                       num_layers=layer, context_dim=context_dim, dropout=dropout)
        
        self.output_gn = nn.GroupNorm(1, self.feature_dim, eps=1e-8)
        self.output = nn.Conv1d(self.feature_dim, self.input_dim * 2 if learn_sigma else self.input_dim, 1)


    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def compress(self, x):
        batchsize, channel, time, freq = x.size()
        x = x.permute(0,1,3,2)
        x = x.reshape(batchsize, -1, time)
        return x
    
    def decompress(self, x):
        batchsize, _, time = x.size()
        x = x.reshape(batchsize, 8, 16, time)
        return x.permute(0,1,3,2)

    def forward(self, x, timesteps, context=None, y=None, autoencoder=None):
        #input = input.to(device)
        if autoencoder is not None:
            x = autoencoder.encode(x)
            x = x.unsqueeze(1)
        
        x = self.compress(x) # Added by haohe
        
        batch_size, input_dim, seq_length = x.shape
        # input: (B, D, T)
        #print(timesteps, flush=True) 
        t_emb = timestep_embedding(timesteps, self.feature_dim, repeat_only=False)
        emb = self.time_embed(t_emb)
        #emb = torch.cat([emb, self.context_embed(context).mean(1)], -1)

        if(self.film_emb is not None):
            # emb = torch.cat([emb, self.film_emb(y)], dim=-1)
            emb = emb + self.film_emb(y)

        x = self.BN(x) # (B, D, L)-->(B, N, L)
        x = self.BN_gn(x)
        #x = F.layer_norm(x, x.shape[1:])
        # split the encoder output into overlapped, longer segments
        x, enc_rest = self.split_feature(x, self.segment_size)  # B, N, L, K: L is the segment_size
        #print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPT
        out = self.DPT(x, emb, context).view(batch_size, self.feature_dim, self.segment_size, -1)  # B, N, L, K

        # overlap-and-add of the outputs
        out = self.merge_feature(out, enc_rest)  # B, N, T
        #out = F.relu(out)
        out = self.output_gn(out)
        out = self.output(out)
        
        if autoencoder is not None:
            out = autoencoder.decode(out)
            if self.predict_xstart:
                out = torch.tanh(out)
    
            return out
        else:
            if self.learn_sigma:
                out = out.view(batch_size, 2, input_dim, seq_length)
            else:
                out = out.view(batch_size, 1, input_dim, seq_length)
        
        if self.predict_xstart:
            out = torch.tanh(out)
        else:
            out = F.layer_norm(out, out.shape[1:])
        
        out = out.squeeze(1)
        out = self.decompress(out) # Added by haohe
        
        return out
    

