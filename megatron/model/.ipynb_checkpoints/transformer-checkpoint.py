# Copyright (c) 2024 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import memory_module
from copy import deepcopy
"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm
from megatron import mpu, print_rank_0
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb,
    AliBi,
)
from megatron.model.fused_rope import (
    FusedRoPEFunc,
    fused_apply_rotary_pos_emb_cached,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention
from deepspeed.moe.layer import MoE
import numpy
import os

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     kv: number of key or value heads
     p: number of model parallel partitions
     np: n/p
     kvp: kv/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        MOE=False,
        MoE_mp_size=1,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = int(4 * 2 / 3) if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * neox_args.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim_in,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            parallel_output=parallel_output,
            skip_bias_add=True,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class NoAddLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight), self.bias
    
class BaselinelMLP(nn.Module):
    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        MOE=False,
        MoE_mp_size=1,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = int(4 * 2 / 3) if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * neox_args.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.hidden_size
        )
        self.dense_h_to_4h = NoAddLinear(neox_args.hidden_size, ff_dim).to(
            device=torch.cuda.current_device(),
            dtype=neox_args.params_dtype
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        self.dense_4h_to_h = NoAddLinear(ff_dim_in, neox_args.hidden_size).to(
            device=torch.cuda.current_device(),
            dtype=neox_args.params_dtype
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Note: multiple_of is used to compute the hidden dimension of the MLP
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
        MOE=False,
        MoE_mp_size=1,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        # Allow custom intermediate size, e.g. for Mistral
        if neox_args.intermediate_size is not None:
            ff_dim = neox_args.intermediate_size
        else:
            ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
            ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w3 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        neox_args,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if neox_args.use_mup = True, despite it not being included here
            )

    #        else:
    #            print(
    #                'ERROR: Output layer parallelism over the hidden dim is currently broken (https://github.com/EleutherAI/gpt-neox/issues/905). Please run with output_layer_parallelism = "column" until this issue is fixed.'
    #            )
    #            exit()
    #            self.final_linear = mpu.RowParallelLinear(
    #                neox_args=neox_args,
    #                input_size=neox_args.hidden_size,
    #                output_size=neox_args.padded_vocab_size,
    #                bias=False,
    #                input_is_parallel=False,
    #                init_method=init_method,
    #                parallel_output=parallel_output,
    #                skip_bias_add=False,
    #                mup_rescale_parameters=is_last_layer,  # only called if neox_args.use_mup = True, despite it not being included here
    #            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb

        self.use_qk_layernorm = neox_args.use_qk_layernorm
        if self.use_qk_layernorm:
            norm, eps = get_norm(neox_args)
            self.qk_layernorm = norm(
                [
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                ],
                eps=eps,
            )

        self.sliding_window_width = neox_args.sliding_window_width

        if (
            not neox_args.num_kv_heads
            or neox_args.num_kv_heads == neox_args.num_attention_heads
        ):
            self.gqa = False
        else:
            self.gqa = True
        if self.gqa:
            self.num_kv_heads_per_partition = mpu.divide(
                neox_args.num_kv_heads, world_size
            )  # we do not yet clone KV heads in MQA across TP ranks...
            self.kv_hidden_size = (
                neox_args.num_kv_heads * self.hidden_size_per_attention_head
            )  # how large the total hidden dim for each of K and V is
        else:
            self.num_kv_heads_per_partition = self.num_attention_heads_per_partition
            self.kv_hidden_size = neox_args.hidden_size

        if not self.gqa:
            # Strided linear layer.
            self.query_key_value = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=3 * neox_args.hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=neox_args.use_bias_in_attn_linear,
            )
        else:
            # QKV proj is smaller if we are using GQA / MQA
            self.query_key_value = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.hidden_size + 2 * self.kv_hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=neox_args.use_bias_in_attn_linear,
            )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        # TODO: this arg shouldn't need to be passed in - get from neox_args
        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim,
                base=neox_args.rotary_emb_base,
                max_seq_len=neox_args.seq_length,
                precision=neox_args.params_dtype,
                save_inv_freqs=neox_args.rotary_save_freqs_buffer,
            )
        else:
            self.rotary_emb = None

        self.rope_fusion = neox_args.rope_fusion
        self.attention_type = neox_args.attention_config[layer_number]
        self.use_flash_attention = self.attention_type == "flash"
        self.sparse = self.attention_type not in ("global", "flash")

        if self.gqa:
            assert not self.sparse

        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            if self.use_flash_attention:
                # we now use Flash Attention 2's provided interface.
                # TODO: we no longer need to use flash_triton_fn since flash cuda supports alibi.
                # consider adding OpenAI's more recent Flash-2 Triton kernel in future
                # from https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
                from flash_attn.flash_attn_interface import (
                    flash_attn_func,
                    flash_attn_varlen_func,
                )
                from flash_attn.flash_attn_triton import (
                    flash_attn_func as flash_attn_unpadded_unpacked_func_triton,
                )

                self.flash_triton_fn = flash_attn_unpadded_unpacked_func_triton
                self.flash_qkv_fn = flash_attn_func
                self.flash_varlen_qkv_fn = flash_attn_varlen_func
            else:
                self.scale_mask_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=self.fp16,
                    input_in_bf16=self.bf16,
                    fusion_type=get_fusion_type(neox_args),
                    mask_func=self.attention_mask_func,
                    softmax_in_fp32=self.attention_softmax_in_fp32,
                    scale=coeff,
                )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.dropout_p = neox_args.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=neox_args.use_bias_in_attn_linear,
        )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if self.pos_emb != "alibi":

            # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
            key_layer = key_layer.transpose(0, 1).reshape(
                output_size[0], output_size[3], self.num_kv_heads_per_partition, -1
            )
            value_layer = value_layer.transpose(0, 1).reshape(
                output_size[0], output_size[3], self.num_kv_heads_per_partition, -1
            )

            batch_size = output_size[0]
            max_seqlen_q = output_size[2]
            max_seqlen_k = output_size[3]

            cu_seqlens_q = torch.arange(
                0,
                (batch_size + 1) * max_seqlen_q,
                step=max_seqlen_q,
                dtype=torch.int32,
                device=query_layer.device,
            )

            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * max_seqlen_k,
                step=max_seqlen_k,
                dtype=torch.int32,
                device=key_layer.device,
            )

            # [sq, b, np, hn] -> [b, sq, np, hn]
            query_layer = query_layer.transpose(0, 1).reshape(
                output_size[0], output_size[2], output_size[1], -1
            )

            # only pass in window_size kwarg to flash-attn
            # if we use Sliding Window Attention.
            # Flash attn defaults to (-1,-1), or
            # does not have this kwarg prior to v2.3.0
            extra_kwargs = (
                {"window_size": (self.sliding_window_width, -1)}
                if self.sliding_window_width is not None
                else {}
            )
            if not self.training:
                q_shape = query_layer.shape
                k_shape = key_layer.shape
                v_shape = value_layer.shape
                is_causal = max_seqlen_q == max_seqlen_k
                output = self.flash_varlen_qkv_fn(
                    query_layer.reshape(
                        (q_shape[0] * q_shape[1], q_shape[2], q_shape[3])
                    ),
                    key_layer.reshape(
                        (k_shape[0] * k_shape[1], k_shape[2], k_shape[3])
                    ),
                    value_layer.reshape(
                        (v_shape[0] * v_shape[1], v_shape[2], v_shape[3])
                    ),
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=None,
                    causal=is_causal,
                    **extra_kwargs,
                )
                output = output.reshape(q_shape)
            else:
                output = self.flash_qkv_fn(
                    query_layer,
                    key_layer,
                    value_layer,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=None,
                    causal=True,
                    **extra_kwargs,
                )

            matmul_result = output
            # [b, sq, np, hn] -> [b, np, sq, hn]
            matmul_result = matmul_result.transpose(1, 2)

        else:
            # [sq, b, np, hn] -> [b, sq, np, hn]
            sq = query_layer.size(0)
            b = query_layer.size(1)
            sk = key_layer.size(0)

            query_layer = query_layer.transpose(0, 1)
            key_layer = key_layer.transpose(0, 1)
            value_layer = value_layer.transpose(0, 1)

            bias = self.alibi_embed.bias(sq, sk, query_layer.device, query_layer.dtype)
            bias = bias.unsqueeze(0).tile((b, 1, 1, 1))

            matmul_result = self.flash_triton_fn(
                query_layer, key_layer, value_layer, bias=bias, causal=True
            )
            matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

    def gqa_project(self, hidden_states, attention_mask, layer_past=None):
        # QKV projection and separation into separate Q/K/V layers for GQA,
        # where KV projections may be smaller than Q projection.
        # the logic for this is explained in comments of this function
        # detailing the intermediate sizes of tensors at each reshape.

        # pass through projection: [sq, b, h] --> [sq, b, ((np + 2 * kvp) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # First: reshape so we have seqlen, batch, and num. query heads each as separate dims
        # Final dim is not exactly head dim: the first (head dim) dims are query heads,
        # The last (head dim * ratio of kv to q heads) each are the "k/v heads"
        # (right now we treat like we have same num. heads, but smaller head dim)

        # [sq, b, ((np + 2 * kvp) * hn)] --> [sq, b, np, (hn * (1 + 2 * (kvp / np)))]
        new_qkv_shape = (
            mixed_x_layer.shape[0],
            mixed_x_layer.shape[1],
            self.num_attention_heads_per_partition,
            int(
                self.hidden_size_per_attention_head
                * (
                    1
                    + 2
                    * (
                        self.num_kv_heads_per_partition
                        / self.num_attention_heads_per_partition
                    )
                )
            ),
        )
        mixed_x_layer = mixed_x_layer.reshape(*new_qkv_shape)

        # Next: split our fake head dim. (last dim) so that the first (head dim) dimensions go to Q,
        # the last smaller 2 * (head dim * kv to q head ratio) each divided between K and V separately
        split_sizes = (
            self.hidden_size_per_attention_head,
            int(
                (
                    self.num_kv_heads_per_partition
                    / self.num_attention_heads_per_partition
                )
                * self.hidden_size_per_attention_head
            ),
            int(
                (
                    self.num_kv_heads_per_partition
                    / self.num_attention_heads_per_partition
                )
                * self.hidden_size_per_attention_head
            ),
        )

        # [sq, b, np, (hn * (1 + 2 * (kvp / np)))] --> 1 x [sq, b, np, hn] , 2 x [sq, b, np, (hn * (kvp / np))]
        (query_layer, key_layer, value_layer) = [
            x.contiguous()
            for x in torch.split(
                mixed_x_layer,
                split_sizes,
                dim=mixed_x_layer.dim() - 1,
            )
        ]

        # reshape K/V to proper output shape (last dim = correct full "real" head size again)
        # 2 x [sq, b, np, (hn * (kvp / np))] --> 2 x [sq, b, kvp, hn]
        new_kv_shape = (
            key_layer.size(0),
            key_layer.size(1),
            self.num_kv_heads_per_partition,
            self.hidden_size_per_attention_head,
        )

        key_layer = key_layer.view(*new_kv_shape)

        value_layer = value_layer.view(*new_kv_shape)

        # if not using Flash attention, we repeat K/V heads to match Q head counts
        if not self.use_flash_attention:
            key_layer = torch.repeat_interleave(
                key_layer,
                repeats=int(
                    self.num_attention_heads_per_partition
                    // self.num_kv_heads_per_partition
                ),
                dim=2,
            )
            value_layer = torch.repeat_interleave(
                value_layer,
                repeats=int(
                    self.num_attention_heads_per_partition
                    // self.num_kv_heads_per_partition
                ),
                dim=2,
            )

        return query_layer, key_layer, value_layer

    def forward(self, hidden_states, attention_mask, layer_past=None):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if not self.gqa:
            # QKV projection for MHA.

            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
        else:
            # Grouped Query Attention (GQA) - specific logic for performing QKV proj
            # and separating out Q, K, and V outputs.

            # output shapes: 1 x [sq, b, np, hn], 2 x [sq, b, kvp, hn] if using flash
            query_layer, key_layer, value_layer = self.gqa_project(
                hidden_states, attention_mask, layer_past=layer_past
            )

        # QK Normalization https://arxiv.org/abs/2302.05442
        if self.use_qk_layernorm:
            query_layer = self.qk_layernorm(query_layer)
            key_layer = self.qk_layernorm(key_layer)

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer

            seq_len = key_layer.shape[0]
            offset = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            if self.rope_fusion:
                query_layer, key_layer = (
                    fused_apply_rotary_pos_emb_cached(rot, cos, sin)
                    for rot in [query_rot, key_rot]
                )
            else:
                if self.bf16:
                    apply_rotary_fn = apply_rotary_pos_emb_torch
                else:
                    apply_rotary_fn = apply_rotary_pos_emb
                query_layer, key_layer = apply_rotary_fn(
                    query_rot, key_rot, cos, sin, offset=offset
                )

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        if self.use_flash_attention:
            context_layer = self.flash_attention(query_layer, key_layer, value_layer)
        elif not self.sparse:
            context_layer = self.attention(
                query_layer, key_layer, value_layer, layer_past, attention_mask
            )
        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, present]

        return output, bias


class MySelfAttention(nn.Module):
    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb

        self.use_qk_layernorm = neox_args.use_qk_layernorm
        if self.use_qk_layernorm:
            norm, eps = get_norm(neox_args)
            self.qk_layernorm = norm(
                [
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                ],
                eps=eps,
            )

        self.sliding_window_width = neox_args.sliding_window_width
        self.gqa = False
        self.num_kv_heads_per_partition = self.num_attention_heads_per_partition
        self.kv_hidden_size = neox_args.hidden_size

        self.hash_proj = None
        if neox_args.attn_memory_config.get("fc_qkv", False):
            print_rank_0('Using nn.Linear for baseline qkv projection')
            self.query_proj = nn.Linear(neox_args.hidden_size, neox_args.hidden_size, bias=neox_args.use_bias_in_attn_linear)
            self.key_proj = nn.Linear(neox_args.hidden_size, neox_args.hidden_size, bias=neox_args.use_bias_in_attn_linear)
            self.value_size = neox_args.attn_memory_config.pop("value_size", neox_args.hidden_size)
            self.value_hidden_size_per_head = mpu.divide(self.value_size, neox_args.num_attention_heads)
            self.value_proj = nn.Linear(neox_args.hidden_size, self.value_size, bias=neox_args.use_bias_in_attn_linear)
            [init_method(self.query_proj.weight), init_method(self.key_proj.weight), init_method(self.value_proj.weight)]

        else:
            layer_cls = memory_module.MemoryLayer

            print_rank_0('Using MemoryLayer for qkv')
            qk_memory_config = deepcopy(neox_args.attn_memory_config)
            qk_memory_config["hidden_size"] = qk_memory_config["output_size"] = neox_args.hidden_size
            self.query_proj = layer_cls(
                qk_memory_config,
                bias=neox_args.use_bias_in_attn_linear)
            self.key_proj = layer_cls(
                qk_memory_config,
                bias=neox_args.use_bias_in_attn_linear)
            
            v_memory_config = deepcopy(neox_args.attn_memory_config)
            v_memory_config["hidden_size"] = neox_args.hidden_size
            self.value_size = v_memory_config.pop("value_size", neox_args.hidden_size)
            v_memory_config["output_size"] = self.value_size
            self.value_hidden_size_per_head = mpu.divide(
                self.value_size, neox_args.num_attention_heads
            )
            self.value_proj = layer_cls(
                v_memory_config,
                bias=neox_args.use_bias_in_attn_linear)
            [init_method(self.query_proj.tables), init_method(self.key_proj.tables), init_method(self.value_proj.tables)]
            
            if neox_args.attn_memory_config["shared_projection"]:
                assert neox_args.attn_memory_config["remove_projection"]
                qkv_table_total_dim = self.value_proj.hash.total_dim
                self.hash_proj = nn.Linear(neox_args.hidden_size, qkv_table_total_dim, bias=True)
                init_method(self.hash_proj)
        
        self.fc_qkv = neox_args.attn_memory_config.get("fc_qkv", False)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        # TODO: this arg shouldn't need to be passed in - get from neox_args
        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim,
                base=neox_args.rotary_emb_base,
                max_seq_len=neox_args.seq_length,
                precision=neox_args.params_dtype,
                save_inv_freqs=neox_args.rotary_save_freqs_buffer,
            )
        else:
            self.rotary_emb = None

        self.rope_fusion = neox_args.rope_fusion
        self.attention_type = neox_args.attention_config[layer_number]
        self.use_flash_attention = self.attention_type == "flash"
        self.sparse = self.attention_type not in ("global", "flash")

        if self.gqa:
            assert not self.sparse

        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            if self.use_flash_attention:
                # we now use Flash Attention 2's provided interface.
                # TODO: we no longer need to use flash_triton_fn since flash cuda supports alibi.
                # consider adding OpenAI's more recent Flash-2 Triton kernel in future
                # from https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
                from flash_attn.flash_attn_interface import (
                    flash_attn_func,
                    flash_attn_varlen_func,
                )
                from flash_attn.flash_attn_triton import (
                    flash_attn_func as flash_attn_unpadded_unpacked_func_triton,
                )

                self.flash_triton_fn = flash_attn_unpadded_unpacked_func_triton
                self.flash_qkv_fn = flash_attn_func
                self.flash_varlen_qkv_fn = flash_attn_varlen_func
            else:
                self.scale_mask_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=self.fp16,
                    input_in_bf16=self.bf16,
                    fusion_type=get_fusion_type(neox_args),
                    mask_func=self.attention_mask_func,
                    softmax_in_fp32=self.attention_softmax_in_fp32,
                    scale=coeff,
                )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.dropout_p = neox_args.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        if neox_args.attn_memory_config["del_dense"]:
            print_rank_0("Del dense in attention module")
            self.dense = lambda x: x
        else:
            self.dense = mpu.RowParallelLinear(
                neox_args=neox_args,
                input_size=self.value_size,
                output_size=neox_args.attn_memory_config["dense_size"],
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=False,
                parallel_output=parallel_output,
                bias=neox_args.use_bias_in_attn_linear,
            )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if self.pos_emb != "alibi":

            # [sk, b, np, hn] -> [b, sk, np, hn] -> [b * sk, 1, np, hn]
            key_layer = key_layer.transpose(0, 1).reshape(
                output_size[0], output_size[3], self.num_kv_heads_per_partition, -1
            )
            value_layer = value_layer.transpose(0, 1).reshape(
                output_size[0], output_size[3], self.num_kv_heads_per_partition, -1
            )

            batch_size = output_size[0]
            max_seqlen_q = output_size[2]
            max_seqlen_k = output_size[3]

            cu_seqlens_q = torch.arange(
                0,
                (batch_size + 1) * max_seqlen_q,
                step=max_seqlen_q,
                dtype=torch.int32,
                device=query_layer.device,
            )

            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * max_seqlen_k,
                step=max_seqlen_k,
                dtype=torch.int32,
                device=key_layer.device,
            )

            # [sq, b, np, hn] -> [b, sq, np, hn]
            query_layer = query_layer.transpose(0, 1).reshape(
                output_size[0], output_size[2], output_size[1], -1
            )

            # only pass in window_size kwarg to flash-attn
            # if we use Sliding Window Attention.
            # Flash attn defaults to (-1,-1), or
            # does not have this kwarg prior to v2.3.0
            extra_kwargs = (
                {"window_size": (self.sliding_window_width, -1)}
                if self.sliding_window_width is not None
                else {}
            )
            if not self.training:
                q_shape = query_layer.shape
                k_shape = key_layer.shape
                v_shape = value_layer.shape
                is_causal = max_seqlen_q == max_seqlen_k
                output = self.flash_varlen_qkv_fn(
                    query_layer.reshape(
                        (q_shape[0] * q_shape[1], q_shape[2], q_shape[3])
                    ),
                    key_layer.reshape(
                        (k_shape[0] * k_shape[1], k_shape[2], k_shape[3])
                    ),
                    value_layer.reshape(
                        (v_shape[0] * v_shape[1], v_shape[2], v_shape[3])
                    ),
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=None,
                    causal=is_causal,
                    **extra_kwargs,
                )
                output = output.reshape(q_shape)
            else:
                output = self.flash_qkv_fn(
                    query_layer,
                    key_layer,
                    value_layer,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=None,
                    causal=True,
                    **extra_kwargs,
                )

            matmul_result = output
            # [b, sq, np, hn] -> [b, np, sq, hn]
            matmul_result = matmul_result.transpose(1, 2)

        else:
            # [sq, b, np, hn] -> [b, sq, np, hn]
            sq = query_layer.size(0)
            b = query_layer.size(1)
            sk = key_layer.size(0)

            query_layer = query_layer.transpose(0, 1)
            key_layer = key_layer.transpose(0, 1)
            value_layer = value_layer.transpose(0, 1)

            bias = self.alibi_embed.bias(sq, sk, query_layer.device, query_layer.dtype)
            bias = bias.unsqueeze(0).tile((b, 1, 1, 1))

            matmul_result = self.flash_triton_fn(
                query_layer, key_layer, value_layer, bias=bias, causal=True
            )
            matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )


    def _forward_memory_qkv(self, hidden_states):
        # Q and K and V share the same hashing code.
        if self.hash_proj is not None:
            hidden_states = self.hash_proj(hidden_states)

        other_shape = hidden_states.shape[:-1]
        hidden_states = hidden_states.reshape(numpy.prod(other_shape).item(), -1)
        code_score = self.value_proj.hash(hidden_states, table_weight=None, force_torch_op=self.value_proj.force_torch_op)
        assert len(code_score) == 2
        code_score_shape = code_score + (other_shape,)
        query_layer = self.query_proj(None, precompute_code_score_shape=code_score_shape)
        key_layer = self.key_proj(None, precompute_code_score_shape=code_score_shape)
        value_layer = self.value_proj(None, precompute_code_score_shape=code_score_shape)
        return query_layer, key_layer, value_layer

    def forward(self, hidden_states, attention_mask, layer_past=None):
        # hidden_states.shape : [sq, b, h]
        new_tensor_shape = hidden_states.size()[:-1] + (
            self.num_attention_heads_per_partition, self.hidden_size_per_attention_head, )
        
        if self.fc_qkv:
            query_layer = self.query_proj(hidden_states)
            key_layer = self.key_proj(hidden_states)
            value_layer = self.value_proj(hidden_states)
        else:
            query_layer, key_layer, value_layer = self._forward_memory_qkv(hidden_states)

        new_value_shape = value_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.value_hidden_size_per_head,)
        
        query_layer = query_layer.view(*new_tensor_shape)
        key_layer = key_layer.view(*new_tensor_shape)
        value_layer = value_layer.view(new_value_shape)

        # QK Normalization https://arxiv.org/abs/2302.05442
        if self.use_qk_layernorm:
            query_layer = self.qk_layernorm(query_layer)
            key_layer = self.qk_layernorm(key_layer)

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer

            seq_len = key_layer.shape[0]
            offset = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            if self.rope_fusion:
                query_layer, key_layer = (
                    fused_apply_rotary_pos_emb_cached(rot, cos, sin)
                    for rot in [query_rot, key_rot]
                )
            else:
                if self.bf16:
                    apply_rotary_fn = apply_rotary_pos_emb_torch
                else:
                    apply_rotary_fn = apply_rotary_pos_emb
                query_layer, key_layer = apply_rotary_fn(
                    query_rot, key_rot, cos, sin, offset=offset
                )

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        if self.use_flash_attention:
            context_layer = self.flash_attention(query_layer, key_layer, value_layer)
        elif not self.sparse:
            context_layer = self.attention(
                query_layer, key_layer, value_layer, layer_past, attention_mask
            )
        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.value_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [sq, b, h]
        tmp_output = self.dense(context_layer) # remove the bias(None) returned by ColumnParallelLinear
        output = tmp_output[0] if isinstance(tmp_output, tuple) else tmp_output

        if self.use_cache:
            output = [output, present]

        return output


class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):

        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)

        # Layernorm on the input data.
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.use_cache = use_cache

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.gpt_j_residual = neox_args.gpt_j_residual
        self.gpt_j_tied = neox_args.gpt_j_tied
        self.mlp_type = neox_args.mlp_type

        if self.gpt_j_residual:
            self.reduce = mpu.mappings.reduce_from_model_parallel_region

        self.aux_loss = neox_args.aux_loss

        self.attention = MySelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            use_cache=self.use_cache,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )
        self.act_fn = memory_module.get_act_fn(neox_args.memory_config["act_fn"])
        self.act_pos = neox_args.memory_config["act_pos"].lower()
        assert self.act_pos in ("after_residual", "before_residual", "none")

        attn_outout_size = self.attention.value_size if neox_args.attn_memory_config["del_dense"] else neox_args.attn_memory_config["dense_size"]
        self.post_attention_layernorm = norm(attn_outout_size, eps=eps)

        self.zero_pad_residual = neox_args.zero_pad_residual
        self.repeat_residual = neox_args.repeat_residual
        assert not (self.zero_pad_residual and self.repeat_residual), 'No residual or either one. Do not turn on both.'
        if self.zero_pad_residual: print_rank_0("use zero_pad_residual ")
        if self.repeat_residual: print_rank_0("use repeat_residual ")
            
        assert neox_args.mlp_type == "MemoryBlock"
        
        # MLP
        def get_mlp(mlp_type, **kw):
            
            if "memory" in mlp_type.lower():

                layer_cls = memory_module.MemoryLayer

                if neox_args.double_mlp_table:
                    assert neox_args.memory_config["remove_projection"]
                    mlp_expand_type = neox_args.mlp_expand_type if neox_args.mlp_expand_type else "extra_bit"
                    assert mlp_expand_type in ("extra_table", "extra_bit")
                    
                    if mlp_expand_type == "extra_bit":
                        extra_bit = 2
                        cfg1 = deepcopy(neox_args.memory_config)
                        cfg1["output_size"] = int(neox_args.memory_config["ffn_num_table"] * (neox_args.memory_config["code_length"] + extra_bit))
                        cfg2 = deepcopy(neox_args.memory_config)
                        cfg2["code_length"] = neox_args.memory_config["code_length"] + extra_bit
                    elif mlp_expand_type == "extra_table":
                        cfg1 = deepcopy(neox_args.memory_config)
                        cfg1["output_size"] = neox_args.memory_config["output_size"] * 2
                        cfg2 = deepcopy(neox_args.memory_config)
                        cfg2["ffn_num_table"] = neox_args.memory_config["ffn_num_table"] * 2
                    else: 
                        raise NotImplementedError("check neox_args.double_mlp_table")

                    return nn.Sequential(
                        layer_cls(cfg1),
                        nn.LayerNorm(cfg1["output_size"]),
                        layer_cls(cfg2)
                    )

                else:
                    return layer_cls(neox_args.memory_config)
            
            elif mlp_type == "regular":
                return ParallelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                    **kw,
                )
            elif mlp_type == "llama":
                return LLaMAParallelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                    **kw,
                )
            elif mlp_type == "baseline":
                return BaselinelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                    **kw,
                )
            else:
                raise KeyError(mlp_type)

        self.num_experts = (
            neox_args.num_experts
            if layer_number % neox_args.expert_interval == 0
            else 1
        )
        args = neox_args
        if self.num_experts <= 1:
            self.mlp = get_mlp(neox_args.mlp_type)
        else:
            from torch import distributed as dist

            if self.num_experts > dist.get_world_size():
                moe_mp_size = 1
            else:
                moe_mp_size = dist.get_world_size() // self.num_experts

            self.mlp = MoE(
                args.hidden_size,
                get_mlp(
                    "regular",
                    MOE=True,
                    MoE_mp_size=moe_mp_size,
                ),
                num_experts=self.num_experts,
                ep_size=args.moe_expert_parallel_size,
                k=args.moe_top_k,
                use_residual=args.moe_use_residual,
                capacity_factor=args.moe_train_capacity_factor,
                eval_capacity_factor=args.moe_eval_capacity_factor,
                min_capacity=args.moe_min_capacity,
                drop_tokens=args.moe_token_dropping,
                use_tutel=args.use_tutel,
            )

        self.layer_past = None  # used to cache k/v pairs in inference

    def forward(self, x, attention_mask, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past
        moe_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        residual = x  # x: [b, s, h]
        # import pdb; pdb.set_trace()
        attention_output = self.attention(
            self.input_layernorm(x),
            attention_mask,
            layer_past=layer_past
        )
        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents
        
        attention_output = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training,)

        # add Residual before layernorm
        # Residual after attn is important for performance
        add_residule_once = False
        last_dim_match = False
        if attention_output.shape[-1] == residual.shape[-1]:
            attention_output = attention_output + residual
            add_residule_once = last_dim_match = True

        elif self.zero_pad_residual:
            dim_diff = attention_output.shape[-1] - residual.shape[-1]
            assert dim_diff  > 0
            _diff_shape = residual.shape[:-1] + (dim_diff,)
            _zero_pad = torch.zeros(_diff_shape, device=residual.device, dtype=residual.dtype)
            residual = torch.cat([residual, _zero_pad], dim=-1)
            attention_output = attention_output + residual
            add_residule_once = True

        elif self.repeat_residual:
            dim_diff = attention_output.shape[-1] - residual.shape[-1]
            assert 0 < dim_diff <= residual.shape[-1], \
                f"dim_diff={dim_diff}, residual.shape[-1]={residual.shape[-1]}, attention_output.shape[-1]={attention_output.shape[-1]}"
            residual = torch.cat([residual, residual[... , : dim_diff]], dim=-1)
            attention_output = attention_output + residual
            add_residule_once = True

        layernorm_output = self.post_attention_layernorm(attention_output)
        mlp_output = self.mlp(layernorm_output)

        #add auxilary loss here by the name "moe_loss": (mlp_output, moe_loss) = self.mlp(...)
        mlp_output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        
        if add_residule_once and last_dim_match:
            output = mlp_output + attention_output
        elif add_residule_once and not last_dim_match:
            output = mlp_output + attention_output[..., : mlp_output.shape[-1]]
            output[..., : dim_diff] = output[..., : dim_diff] + attention_output[..., mlp_output.shape[-1] : ]
        else:
            output = mlp_output + residual

        return output, moe_loss


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        output, moe_loss = super().forward(hidden_states, attention_mask)
        # auxiliary output
        self.last_moe_loss = moe_loss
        return output, attention_mask


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        logits, bias = super().forward(hidden_state)
        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.norm(args)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
