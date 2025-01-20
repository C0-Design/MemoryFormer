import torch.nn as nn
import torch
import math
import numpy as np
from .compute_code_score.kernel import compute_code_score
from .gather.kernel import gather

import os

def get_act_fn(fn):
    fn = fn.lower()
    if fn == "silu":
        return nn.functional.silu
    elif fn == "gelu":
        return nn.functional.gelu
    elif fn == "softmax":
        return nn.Softmax(dim=-1)
    elif fn == "tanh":
        return torch.tanh
    elif fn == "sigmoid":
        return torch.sigmoid
    else:
        raise NotImplementedError(f'No such activation {fn}')


class Hashing(nn.Module):
    def __init__(self, memory_cfg):
        super().__init__()

        self.projection_type = projection_type = memory_cfg["projection_type"].lower()
        block_size = memory_cfg["block_size"]

        self.hidden_size = memory_cfg["hidden_size"]
        self.num_table = memory_cfg["ffn_num_table"]
        self.code_length = memory_cfg["code_length"]
        self.total_dim = self.num_table * self.code_length
        
        self.remove_projection = memory_cfg["remove_projection"]
        if self.remove_projection:
            print("Removing hashing projection in MemoryLayer!")
        elif projection_type == "dense":
            self.projection = nn.Linear(self.hidden_size, self.total_dim)
        else:
            raise NotImplementedError
            
        self.act_fn = get_act_fn(memory_cfg["act_fn"])

    def forward(self, x, table_weight=None, force_torch_op=False):

        if self.remove_projection:
            z = x
        else:
            z = self.projection(x)
                    
        B, D = z.shape
        assert D == self.total_dim, f"expect the last dim = {self.total_dim}, but actually go {D}"
        z = z.reshape(B, self.num_table, self.code_length)
        # import pdb; pdb.set_trace()
        code, score, self.last_z_abs = compute_code_score(z, force_torch_op=force_torch_op)

        return code, score


class MemoryLayer(nn.Module):

    def __init__(
            self, 
            memory_cfg,
            force_torch_op=False,
            skip_bias_add_and_return=False,
            bias=True
        ):
        super().__init__()

        check_args(memory_cfg)
        hidden_size = memory_cfg["hidden_size"]
        output_size = memory_cfg["output_size"]

        self.skip_bias_add_and_return = skip_bias_add_and_return
        self.force_torch_op = force_torch_op
        if self.force_torch_op:
            print('force_torch_op only set to True in test and profiling!')
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_table = memory_cfg["ffn_num_table"]
        self.code_length = memory_cfg["code_length"]
        self.table_size = int(2 ** self.code_length)

        self.hash = Hashing(memory_cfg)
        self.tables = nn.Parameter(0.02 * torch.randn(int(self.num_table*self.table_size), self.output_size))
        self.bias = nn.Parameter(0.001 * torch.randn(self.output_size))

    def extra_repr(self):
        strs = []
        strs.append(f"hidden_size={self.hidden_size}, num_table={self.num_table}, code_length={self.code_length}, table_size={self.table_size}, output_size={self.output_size}")
        return "\n".join(strs)

    def forward(self, hidden_states, table_weight=None, precompute_code_score_shape=None):

        if precompute_code_score_shape is not None:
            # for QKV, Hashing.forward only needs to be called once.
            assert hidden_states is None
            code, score, other_shape = precompute_code_score_shape
        else:
            # (bs, seq_len, dim) = hidden_states.shape
            other_shape = hidden_states.shape[:-1]
            hidden_states = hidden_states.reshape(np.prod(other_shape).item(), -1)
            code, score = self.hash(hidden_states, table_weight=table_weight, force_torch_op=self.force_torch_op)

        if self.force_torch_op:
            outputs = self.gather_compute_flops(code, score)
        else:
            outputs = self.gather(code, score)

        outputs = outputs.reshape(*other_shape, self.output_size)
        
        if self.skip_bias_add_and_return:
            return (outputs, self.bias)
        else:
            return outputs + self.bias         
    
    def gather(self, indexes, weights):
        """ ==========================================
            This torch implementation is mathematically equal.
            It's used for measuring FLOPs but large batch_size causes CUDA_OOM.
        ==============================================
        outputs = my_torch_gather(
            indexes, 
            self.tables.reshape(self.num_table, self.table_size, self.output_size),
            self.hidden_size
        )
        outputs = outputs * weights.unsqueeze(-1)
        return outputs.sum(dim=1)
        """
        B, N = indexes.shape
        assert weights.shape[0] == B and weights.shape[1] == N

        indexes = indexes + torch.arange(self.num_table, device = indexes.device, dtype = indexes.dtype)[None] * self.table_size
        outputs = gather(
            indexes, 
            weights,
            self.tables,
        )
        return outputs

    def gather_compute_flops(self, indexes, weights):
        outputs = my_torch_gather(
            indexes, 
            self.tables.reshape(self.num_table, self.table_size, self.output_size),
            self.output_size
        )
        outputs = outputs * weights.unsqueeze(-1)
        return outputs.sum(dim=1)


def my_torch_gather(indexes, tables, output_size):
    indexes = indexes.unsqueeze(-1).expand(*indexes.shape, output_size).long()
    # tables.shape becomes [table_size, table_num, output_size] after permute
    outputs = tables.permute(1,0,2).gather(0, indexes)
    """return.shap == [B, num_table, output_size]"""
    return outputs


def check_args(config):
    assert isinstance(config["hidden_size"], int)
    assert isinstance(config["output_size"], int)
    assert isinstance(config["ffn_num_table"], int)
    assert isinstance(config["code_length"], int)
    assert isinstance(config["block_size"], int)
    assert isinstance(config["remove_projection"], bool)
    assert isinstance(config["act_fn"], str)
