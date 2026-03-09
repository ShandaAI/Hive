import ipdb
import copy
import math
import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange
from typing import Optional, Tuple
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.utils import weight_norm, remove_weight_norm
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from latent_diffusion.modules.nn import avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module, checkpoint


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d



class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        #self.offset = nn.Parameter(torch.zeros(d))
        #self.register_parameter("offset", self.offset)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed# + self.offset


class BiLSTMproj(nn.Module):

    def __init__(self, enc_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.
        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        super().__init__()
        # Bi-LSTM with learnable (h_0, c_0) state
        self.rnn = nn.LSTM(enc_dim, enc_dim,
                           1, dropout=0, batch_first=True, bidirectional=True, bias=True)
        self.cell_init = nn.Parameter(torch.rand(1, 1, enc_dim))
        self.hidden_init = nn.Parameter(torch.rand(1, 1, enc_dim))

        # Linear projection layer
        self.proj = nn.Linear(enc_dim * 2, enc_dim, bias=False)

    def forward(self, intra_segs):
        """
        Process through a locally recurrent layer along the intra-segment
            direction.
        Parameters:
        	frames (tensor): A batch of intra-segments in shape `[B*S, K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            lr_output (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size_seq_len = intra_segs.size(0)
        cell = self.cell_init.repeat(2, batch_size_seq_len, 1)
        hidden = self.hidden_init.repeat(2, batch_size_seq_len, 1)
        rnn_output, _ = self.rnn(intra_segs, (hidden, cell))
        lr_output = self.proj(rnn_output)
        return lr_output


class BiGRUproj(nn.Module):

    def __init__(self, enc_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.
        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        super().__init__()
        # Bi-LSTM with learnable (h_0, c_0) state
        self.rnn = nn.GRU(enc_dim, enc_dim,
                          1, dropout=0, batch_first=True, bidirectional=True)
        self.cell_init = nn.Parameter(torch.rand(1, 1, enc_dim))

        # Linear projection layer
        self.proj = nn.Linear(enc_dim * 2, enc_dim)

    def forward(self, intra_segs):
        """
        Process through a locally recurrent layer along the intra-segment
            direction.
        Parameters:
        	frames (tensor): A batch of intra-segments in shape `[B*S, K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            lr_output (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size_seq_len = intra_segs.size(0)
        cell = self.cell_init.repeat(2, batch_size_seq_len, 1)
        rnn_output = self.rnn(intra_segs, cell)[0]
        lr_output = self.proj(rnn_output)
        return lr_output


class BiSRUproj(nn.Module):

    def __init__(self, enc_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.
        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        from sru import SRU
        super().__init__()
        self.rnn = SRU(enc_dim, enc_dim, 2, dropout=0.2, bidirectional=True, rescale=True)
        self.cell_init = nn.Parameter(torch.rand(2, 1, enc_dim * 2))

        # Linear projection layer
        self.proj = nn.Linear(enc_dim * 2, enc_dim)

    def forward(self, intra_segs):
        """
        Process through a locally recurrent layer along the intra-segment
            direction.
        Parameters:
        	frames (tensor): A batch of intra-segments in shape `[B*S, K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            lr_output (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size_seq_len = intra_segs.size(0)
        cell = self.cell_init.repeat(1, batch_size_seq_len, 1)
        rnn_output = self.rnn(intra_segs.transpose(0, 1).contiguous(), cell)[0]
        lr_output = self.proj(rnn_output.transpose(0, 1).contiguous())
        return lr_output


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(0, seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class RoFormerSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.num_attention_heads = nhead
        self.attention_head_size = int(d_model / nhead)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(d_model, self.all_head_size, bias=False)
        self.key = nn.Linear(d_model, self.all_head_size, bias=False)
        self.value = nn.Linear(d_model, self.all_head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        sinusoidal_pos=None,
        sinusoidal_pos_context=None,
        context=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # rotary query
        query_layer = self.apply_rotary(query_layer, sinusoidal_pos)

        if context is not None:
            key_layer = self.transpose_for_scores(self.key(context))
            value_layer = self.transpose_for_scores(self.value(context))

            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos_context)
            value_layer = self.apply_rotary(value_layer, sinusoidal_pos_context)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            value_layer = self.apply_rotary(value_layer, sinusoidal_pos)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->RoFormer
class RoFormerSelfOutput(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model, bias=True)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class RoFormerAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.self = RoFormerSelfAttention(d_model, nhead, dim_feedforward, dropout)
        self.output = RoFormerSelfOutput(d_model, nhead, dim_feedforward, dropout)
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    # End Copy
    def forward(
        self,
        hidden_states,
        sinusoidal_pos=None,
        sinusoidal_pos_context=None,
        context=None,
    ):
        self_outputs = self.self(
            hidden_states,
            sinusoidal_pos,
            sinusoidal_pos_context,
            context
        )
        outputs = self.output(self_outputs, hidden_states)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->RoFormer
class RoFormerIntermediate(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.dense = nn.Linear(d_model, dim_feedforward, bias=True)
        self.intermediate_act_fn = NewGELUActivation()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RoFormer
class RoFormerOutput(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.dense = nn.Linear(dim_feedforward, d_model, bias=True)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class RoFormerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0, max_position_embeddings=1000):
        super().__init__()
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            max_position_embeddings,
            d_model // nhead,
        )
        self.attention = RoFormerAttention(d_model, nhead, dim_feedforward, dropout)
        #self.crossattention = RoFormerAttention(d_model, nhead, dim_feedforward, dropout)
        self.intermediate = RoFormerIntermediate(d_model, nhead, dim_feedforward, dropout)
        self.output = RoFormerOutput(d_model, nhead, dim_feedforward, dropout)

    def forward(
        self,
        hidden_states,
        sinusoidal_pos=None,
        context=None,
    ):
        # [sequence_length, embed_size_per_head] -> sin & cos [batch_size, num_heads, sequence_length, embed_size_per_head // 2]
        sinusoidal_pos = self.embed_positions(hidden_states.shape[1])[
            None, None, :, :
        ].chunk(2, dim=-1)

        if context is None:
            self_attention_outputs = self.attention(
                hidden_states,
                sinusoidal_pos,
            )
            attention_output = self_attention_outputs

        else:
            sinusoidal_pos_context = self.embed_positions(context.shape[1])[
                None, None, :, :
            ].chunk(2, dim=-1)
            cross_attention_outputs = self.attention(
                hidden_states,
                sinusoidal_pos,
                sinusoidal_pos_context,
                context
            )
            attention_output = cross_attention_outputs

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SequenceProcessingBlock(nn.Module):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
    """
    def __init__(self, input_size, dropout, method='rnn', cross_attn=False):
        super(SequenceProcessingBlock, self).__init__()
        self.method = method
        self.cross_attn = cross_attn
        if method == 'lstm':
            self.seq_model = BiLSTMproj(input_size)
        elif method == 'gru':
            self.seq_model = BiGRUproj(input_size)
        elif method == 'sru':
            self.seq_model = BiSRUproj(input_size)
        elif method == 'roformer':
            self.seq_model = RoFormerLayer(input_size, 12, input_size * 4, dropout=dropout)
        self.dropout = Dropout(dropout)
        self.out_norm = RMSNorm(input_size)
        self.act_fn = NewGELUActivation()
        if cross_attn:
            self.emb_layers = RoFormerLayer(input_size, 12, input_size * 4, dropout=dropout)
        else:
            self.emb_layers = nn.Sequential(
                self.act_fn,
                Linear(
                    input_size,
                    2 * input_size,
                ),
            )
        self.mod_norm = RMSNorm(input_size)
        self.out_layers = nn.Sequential(
	    self.act_fn,
            nn.Conv1d(input_size, input_size, 7, padding=3, bias=False)
        )

    def forward(self, x, emb):
        # input shape: batch, seq, dim
        x = self.seq_model(x)
        if self.cross_attn:
            x = self.mod_norm(x + self.dropout(self.emb_layers(x, context=emb)))
        else:
            emb_out = self.emb_layers(emb)
            if emb_out.shape[1] != 1:
                emb_out = emb_out.mean(1, keepdim=True)
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            x = self.mod_norm(x * (1 + scale) + shift)
        x = self.out_norm(x + self.dropout(self.out_layers(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()))
        return x


# dual-path blocks
class DualPathBlocks(nn.Module):

    def __init__(self, input_size, segment_size, output_size,
            intra_seq2seq='lstm', inter_seq2seq='lstm',
            num_layers=1, context_dim=512, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.segment_size = segment_size
        self.num_layers = num_layers

        # dual-path fine-coarse models
        self.row_fine_model = nn.ModuleList([])
        self.col_coarse_model = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_fine_model.append(SequenceProcessingBlock(input_size, dropout, method=intra_seq2seq))
            self.col_coarse_model.append(SequenceProcessingBlock(input_size, dropout, method=inter_seq2seq, cross_attn=True))
            self.row_norm.append(nn.GroupNorm(2, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(2, input_size, eps=1e-8))

        # output layer
        self.act_fn = NewGELUActivation()
        self.output = nn.Sequential(self.act_fn, nn.Conv2d(input_size, output_size, 1))

    def forward(self, x, glb_emb, seq_emb, w2v_cfg=False):
        # input shape: batch, N, dim1, dim2
        # apply transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        batch_size, D, dim1, dim2 = x.shape
        output = x
        upper = []
        for i in range(self.num_layers):
            row_input = output.permute(0, 3, 2, 1).reshape(batch_size * dim2, dim1, D)  # B*dim2, dim1, N
            row_output = self.row_fine_model[i](row_input, glb_emb.repeat_interleave(dim2, 0))
            row_output = row_output.view(batch_size, dim2, dim1, D).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 3, 1, 2).reshape(batch_size * dim2, D, dim1)
            merge_scale = 2 ** min(i+1, self.num_layers-i)
            col_input = F.avg_pool1d(col_input, kernel_size=merge_scale * 2, stride=merge_scale, padding=merge_scale//2)
            chunk_size = col_input.shape[-1]
            col_input = col_input.view(batch_size, dim2, D, chunk_size).permute(0, 2, 3, 1).reshape(batch_size * chunk_size, dim2, D)
            col_output = self.col_coarse_model[i](col_input, col_input if w2v_cfg else seq_emb.repeat_interleave(chunk_size, 0))
            if merge_scale == 2 ** (i + 1):
                upper.append(col_output)
            else:
                col_output = col_output + upper[-1]
                del upper[-1]
            col_output = col_output.view(batch_size, chunk_size, dim2, D).permute(0, 2, 3, 1).reshape(batch_size * dim2, D, chunk_size)
            col_output = col_output.repeat_interleave(merge_scale, -1)
            col_output = col_output.view(batch_size, dim2, D, dim1).permute(0, 2, 3, 1).reshape(batch_size, D, dim1, dim2)
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        output = self.output(output) # B, output_size, dim1, dim2

        return output


# base module for deep DPT
class DPRTNet(nn.Module):
    def __init__(self,
            input_dim=256,
            feature_dim=1024,
            layer=8,
            segment_size=64,
            context_dim=512,
            dropout=0,
            diffusion_steps=100,
            intra_seq2seq='lstm',
            inter_seq2seq='lstm',
            predict_xstart=False,
            learn_sigma=False,
            end2end=False,
        ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.diffusion_steps = diffusion_steps

        self.layer = layer
        self.segment_size = segment_size
        self.predict_xstart = predict_xstart
        self.learn_sigma = learn_sigma

        self.eps = 1e-8

        self.act_fn = NewGELUActivation()
        self.time_slerp_points = nn.Embedding(2, 256)
        self.time_embed = nn.Sequential(
            nn.Linear(256, feature_dim),
            self.act_fn,
            Linear(feature_dim, feature_dim),
            RMSNorm(feature_dim)
        )
        if self.context_dim == 129:  # Only happens with w2v-bert + mulan tokens
            self.context_embed = nn.Sequential(
                nn.Embedding(1024, feature_dim),
                self.act_fn,
                Linear(feature_dim, feature_dim),
                RMSNorm(feature_dim)
            )
            self.context_embed_mulan = nn.Sequential(
                nn.Dropout(dropout),
                zero_module(Linear(128, feature_dim)),
                self.act_fn,
                Linear(feature_dim, feature_dim),
                RMSNorm(feature_dim)
            )
        else:
            if context_dim == 1:
                emb_first_layer = nn.Embedding(1024, feature_dim)
            else:
                emb_first_layer = Linear(context_dim, feature_dim)
            self.context_embed = nn.Sequential(
                emb_first_layer,
                self.act_fn,
                Linear(feature_dim, feature_dim),
                RMSNorm(feature_dim)
            )
        # bottleneck
        self.input_map = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)
        self.input_norm = RMSNorm(self.feature_dim)

        # DPT model
        self.blocks = DualPathBlocks(self.feature_dim, self.segment_size, self.feature_dim,
                                     intra_seq2seq=intra_seq2seq, inter_seq2seq=inter_seq2seq,
                                     num_layers=layer, context_dim=context_dim, dropout=dropout)
        
        self.output_norm = RMSNorm(self.feature_dim)
        self.output = nn.Conv1d(self.feature_dim, self.input_dim, 1, bias=False)
        self.autoencoder = None

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder
        for p in self.autoencoder.encoder.parameters():
            p.requires_grad = False
        for p in self.autoencoder.mean_logvar_conv.parameters():
            p.requires_grad = False
        for p in self.autoencoder.decoder.parameters():
            p.requires_grad = False

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

    def forward(self, x, timesteps=None, context=None, encode_only=False, mulan_cfg=False, w2v_cfg=False):
        #input = input.to(device)
        in_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)
        if self.autoencoder is not None:
            x = self.autoencoder.encode(x)
            if encode_only:
                return x
        batch_size, input_dim, seq_length = x.shape
        # input: (B, D, T)
        #print(timesteps[0], flush=True)
        #t_emb = timestep_embedding(timesteps, self.feature_dim, repeat_only=False)
        t_start_emb = self.time_slerp_points(torch.zeros(1, device=x.device, dtype=torch.long))
        t_end_emb = self.time_slerp_points(torch.ones(1, device=x.device, dtype=torch.long))
        low_norm = t_start_emb/torch.norm(t_start_emb, dim=1, keepdim=True)
        high_norm = t_end_emb/torch.norm(t_end_emb, dim=1, keepdim=True)
        omega = torch.acos((low_norm*high_norm).sum(1))
        so = torch.sin(omega)
        t_emb = (torch.sin((1.0-timesteps[:, None])*omega) / so) * t_start_emb\
            + (torch.sin(timesteps[:, None]*omega) / so) * t_end_emb
        t_emb = self.time_embed(t_emb).unsqueeze(1)
        #if context is None:
        glb_emb = t_emb
        seq_emb = t_emb
        if self.training:
            if np.random.rand() < 0.1:
                mulan_cfg = True
            if np.random.rand() < 0.1:
                w2v_cfg = True
        if context is not None:
            if self.context_dim == 129:  # Only happens with w2v-bert + mulan tokens
                mulan_emb = context[..., :128].view(batch_size, 1, 128).float()
                mulan_emb = self.context_embed_mulan(mulan_emb)
                w2v_tokens = context[..., 128:].view(batch_size, -1).long()
                w2v_emb = self.context_embed(w2v_tokens)
                if not mulan_cfg:
                    glb_emb = glb_emb + mulan_emb
                else:
                    if self.training:
                        glb_emb = glb_emb + 0. * mulan_emb
                if not w2v_cfg:
                    seq_emb = seq_emb + w2v_emb
                else:
                    if self.training:
                        glb_emb = glb_emb + 0. * w2v_emb
            else:
                if self.context_dim == 1:
                    context = context.view(batch_size, -1).long()
                glb_emb = glb_emb + self.context_embed(context.detach())
                seq_emb = glb_emb

        x = self.input_map(x) # (B, D, L)-->(B, N, L)
        x = self.input_norm(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        # split the encoder output into overlapped, longer segments
        x, enc_rest = self.split_feature(x, self.segment_size)  # B, N, L, K: L is the segment_size
        #print('enc_segments.shape {}'.format(x.shape))
        out = self.blocks(x, glb_emb, seq_emb, w2v_cfg).view(batch_size, self.feature_dim, self.segment_size, -1)  # B, N, L, K

        # overlap-and-add of the outputs
        out = self.merge_feature(out, enc_rest)  # B, N, T
        out = self.output_norm(out.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        out = self.output(out)
         
        out = out.view(batch_size, input_dim, seq_length)
        
        if self.predict_xstart:
            out = torch.clamp(out, -1, 1)
            #out = torch.tanh(out)
        if self.autoencoder is not None and not self.training:
            wav = self.autoencoder.decode(out).squeeze(1)
            return wav
        return out

