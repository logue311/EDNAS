from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    #'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 8
CONCAT = 8

EDNAS = Genotype(recurrent=[('identity', 0), ('tanh', 1), ('relu', 1), ('tanh', 2), ('relu', 4), ('relu', 3), ('sigmoid', 4), ('tanh', 7)], concat=range(1, 9))