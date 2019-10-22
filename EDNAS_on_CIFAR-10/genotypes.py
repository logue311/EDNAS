from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

EDNAS = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_5x5', 0), ('skip_connect', 4)], reduce_concat=[2, 3, 4, 5])