from torch.utils.cpp_extension import load
from .cuda.kernel import weighted_gather_add as weighted_gather_add_cuda
from .cpu.kernel import weighted_gather_add as weighted_gather_add_cpu


def gather(indices, weights, tables):
    use_cuda = indices.is_cuda and weights.is_cuda and tables.is_cuda
    if use_cuda:
        return weighted_gather_add_cuda(indices, tables, weights)
    else:
        assert (not indices.is_cuda) and (not weights.is_cuda) and (not tables.is_cuda), \
            'All tensors must be on the same device. All on cpu or all on cuda'
        return weighted_gather_add_cpu(indices, tables, weights)
        
    # if training:
    #     assert indices.is_cuda and weights.is_cuda and tables.is_cuda
    #     return weighted_gather_add_cuda(indices, tables, weights)
    # else:
    #     if indices.is_cuda:
    #         assert weights.is_cuda and tables.is_cuda
    #         return weighted_gather_add_cuda(indices, tables, weights)
    #     else:
    #         assert (not weights.is_cuda) and (not tables.is_cuda)
    #         return weighted_gather_add_cpu(indices, tables, weights)
