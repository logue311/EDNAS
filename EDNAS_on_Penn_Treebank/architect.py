import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class Architect(object):

  def __init__(self, model, args):
    self.network_weight_decay = args.wdecay
    self.network_clip = args.clip
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay)
    self.reward_c = 80.0

  def step(self,          
          hidden_valid, input_valid, target_valid, is_opt_step):    
    
    hidden, loss = self._backward_step_policy_rl(hidden_valid, input_valid, target_valid)
    
    if is_opt_step:
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    return hidden, loss

  def _backward_step_policy_rl(self, hidden_valid, input_valid, target_valid):
    loss, hidden_next = self.model._loss(hidden_valid, input_valid, target_valid)
    loss = loss.detach()
    ppl = torch.exp(loss)
    
    reward = self.reward_c / ppl
    #print (reward)
    sample_log_probs = self.model.sample_log_probs
    
    self.model.baseline = self.model.baseline - (1 - self.model.bl_dec) * (self.model.baseline - reward)
    arch_loss = sample_log_probs * (reward - self.model.baseline)
    arch_loss.backward()
    
    return hidden_next, loss