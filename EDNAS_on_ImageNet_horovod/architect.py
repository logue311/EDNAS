import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.lr = args.arch_learning_rate    

  def step(self, input_valid, target_valid, is_opt_step):
    if is_opt_step:
        self.optimizer.zero_grad()
    
    self._backward_step_policy_rl(input_valid, target_valid)
    
    if is_opt_step:
        self.optimizer.step()



