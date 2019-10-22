import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    probs = F.softmax(self.weights, dim=-1)

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid)
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if torch.sum(probs[offset:offset+i+1, k]).data > 0:
          fn = self._get_activation(name)
          unweighted = states + c * (fn(h) - states)
          s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
      s = self.bn(s)
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()
        self.baseline = Variable(torch.zeros(1), requires_grad=False).cuda()
        self.bl_dec = 0.99

    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
      k = sum(i for i in range(1, STEPS+1))
      #weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3)
      #self.weights = Variable(weights_data.cuda(), requires_grad=True)
      self.weights = Variable(torch.zeros(k, len(PRIMITIVES)).cuda(), requires_grad=False)
      self.edge_weights = []      
      for i in range(STEPS-1):
        #self.edge_weights.append(Variable(1e-3*torch.randn(i+2).cuda(), requires_grad=True))        
        self.edge_weights.append(Variable(torch.zeros(i+2).cuda(), requires_grad=False))
        
      self._arch_parameters = [self.weights]
      self._arch_parameters.extend(self.edge_weights)
      #for rnn in self.rnns:
        #rnn.weights = self.weights

    def sample_new_architecture(self):
      selected_edges = [0]    
      criterion = nn.CrossEntropyLoss()
      criterion = criterion.cuda()
      log_prob_list = []
      for i in self.edge_weights:
        edge_probs = F.softmax(i, dim=-1)
        sampled = torch.multinomial(edge_probs, 1)
        log_prob_list.append(criterion(edge_probs.view(1, edge_probs.size(0)), sampled))
        selected_edges.append(int(sampled))    
    
      k = sum(i for i in range(1, STEPS+1))
      num_ops = len(PRIMITIVES)
      self.sampled_weight = Variable(1e-3*torch.zeros(k, num_ops).cuda(), requires_grad=False)
      
      offset = 0
      for i in range(len(selected_edges)):
        edge_idx = selected_edges[i]+offset
        op_weights = self.weights[edge_idx]
        op_probs = F.softmax(op_weights, dim=-1)
        sampled = torch.multinomial(op_probs, 1)
        log_prob_list.append(criterion(op_probs.view(1, op_probs.size(0)), sampled))
        self.sampled_weight[edge_idx, int(sampled)] = 1.0
        offset += i + 1
      
      for rnn in self.rnns:
        rnn.weights = self.sampled_weight
      self.sample_log_probs = torch.sum(torch.stack(log_prob_list))
        
    def arch_parameters(self):
      return self._arch_parameters

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next

    def genotype(self):

      def _parse(probs):
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[0]
          k_best = None
          for k in range(len(W[j])):            
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          start = end
        return gene

      gene = _parse(self.sampled_weight.data.cpu().numpy())
      genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
      return genotype

