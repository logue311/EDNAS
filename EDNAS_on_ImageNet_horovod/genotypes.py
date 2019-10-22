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

EDNAS = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 1), ('max_pool_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=[2, 3, 4, 5])