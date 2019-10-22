import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import random

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    #return sum(w * op(x) for w, op in zip(weights, self._ops))
    for w, op in zip(weights, self._ops):      
      if w.data > 0.0:
        return w * op(x)
    return w     


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    print(C_prev_prev, C_prev, C)
    #Added
    self.reduction_prev = reduction_prev

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)    
    
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):      
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=5, multiplier=5, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    #Added
    self.fixed_edges = None
    self.baseline = Variable(torch.zeros(1), requires_grad=False).cuda()
    self.bl_dec = 0.99
    
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2, affine=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C, affine=False),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C, affine=False),
    )

    C_prev_prev, C_prev, C_curr = C, C, C 


    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell._multiplier * C_curr


    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    self.edge_comb_dict = {}    
    for i in range(self._steps):
      count = 0
      temp_dict = {}
      for j in range(i+2):
        for k in range(j+1):
          temp_dict[count] = (k, j+1)
          count += 1
      self.edge_comb_dict[i] = temp_dict

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
#     s0 = s1 = self.stem(input)
    s0 = self.stem0(input)
    s1 = self.stem1(s0)

    for i, cell in enumerate(self.cells):
      if cell.reduction:
        #weights = F.softmax(self.alphas_reduce, dim=-1)
        weights = self.sampled_weight_reduce        
        #weights = reduce_weights
      else:
        #weights = F.softmax(self.alphas_normal, dim=-1)
        weights = self.sampled_weight_normal
        #weights = normal_weights
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    #self.alphas_normal = Variable(1e-3*torch.zeros(k, num_ops).cuda(), requires_grad=False)
    #self.alphas_reduce = Variable(1e-3*torch.zeros(k, num_ops).cuda(), requires_grad=False)    
    self.edge_alphas_normal = []
    self.edge_alphas_reduce = []
    for i in range(self._steps-1):
      self.edge_alphas_normal.append(Variable(1e-3*torch.randn(sum(j for j in range(i+3))).cuda(), requires_grad=True))
      self.edge_alphas_reduce.append(Variable(1e-3*torch.randn(sum(j for j in range(i+3))).cuda(), requires_grad=True))
    
    self._arch_parameters = [      
      self.alphas_normal,
      self.alphas_reduce,      
    ]
    self._arch_parameters.extend(self.edge_alphas_normal)
    self._arch_parameters.extend(self.edge_alphas_reduce)

  def sample_new_architecture(self):
    selected_normal_edges = [0, 1]
    selected_reduce_edges = [0, 1]
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    log_prob_list = []
    for i in range(len(self.edge_alphas_normal)):
      comb_dict = self.edge_comb_dict[i]
      for j in range(2):
        if j == 0:
          edges = self.edge_alphas_normal[i]
        else:
          edges = self.edge_alphas_reduce[i]
        edge_probs = F.softmax(edges, dim=-1)
        sampled = torch.multinomial(edge_probs, 1)
        log_prob_list.append(criterion(edge_probs.view(1, edge_probs.size(0)), sampled))
        edge_comb = int(sampled)
        edge1, edge2 = comb_dict[edge_comb]
        
        if j == 0:
          selected_normal_edges.append(edge1)
          selected_normal_edges.append(edge2)
        else:
          selected_reduce_edges.append(edge1)
          selected_reduce_edges.append(edge2)
    
    
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.sampled_weight_normal = Variable(1e-3*torch.zeros(k, num_ops).cuda(), requires_grad=False)
    self.sampled_weight_reduce = Variable(1e-3*torch.zeros(k, num_ops).cuda(), requires_grad=False)    
    for j in range(2):
      offset = 0
      for i in range(self._steps):
        for e in range(2):          
          if j == 0:
            edge_index = offset+selected_normal_edges[i*2+e]
            op_weights = self.alphas_normal[edge_index]
          else:
            edge_index = offset+selected_reduce_edges[i*2+e]
            op_weights = self.alphas_reduce[edge_index]
          op_probs = F.softmax(op_weights, dim=-1)
          sampled = torch.multinomial(op_probs, 1)
          log_prob_list.append(criterion(op_probs.view(1, op_probs.size(0)), sampled))
          if j == 0:
            self.sampled_weight_normal[edge_index, int(sampled)] = 1.0
          else:
            self.sampled_weight_reduce[edge_index, int(sampled)] = 1.0
        offset += (i+2)
          
    self.sample_log_probs = torch.sum(torch.stack(log_prob_list))  
    
  def arch_parameters(self):
    return self._arch_parameters

  def named_arch_parameters(self):
    return [(str(step), params) for step, params in enumerate(self._arch_parameters)]

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):      
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    #gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    #gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    
    normal_weights, reduce_weights = self._compute_weight_from_alphas()    
    gene_normal = _parse(normal_weights.data.cpu().numpy())
    gene_reduce = _parse(reduce_weights.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  