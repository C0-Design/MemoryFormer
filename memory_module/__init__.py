from .memory import MemoryLayer, get_act_fn
from argparse import Namespace

memory_config = Namespace(
    hidden_size = 512, 
    output_size = 512, 
    ffn_num_table = 128, 
    code_length = 8,
    ffn_table_size = 256,
    projection_type = "dense",  # "bh4"
    block_size = 64,
)