import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pickle
import time
import pandas as pd
import random
import math

from torch.autograd import Variable
from model_search import Network
from architect import Architect

from genotypes import PRIMITIVES
from genotypes import Genotype

import warnings
warnings.filterwarnings('ignore')

import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod.torch as hvd
import tensorboardX
from tqdm import tqdm

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3.5e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
#Added
parser.add_argument('--max_hidden_node_num', type=int, default=1, help='max number of hidden nodes in a cell')

#horovod 
parser.add_argument('--batch-size', type=int, default=200,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=50,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.05,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--wd', type=float, default=0.00003,
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./{exp}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(args.save):
    os.makedirs(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CLASSES = 1000
node_num = 5
search_epoch = 1
search_arch_num = 100
arch_opt_step = 1
top_arch_num = 10

def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_queue)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 10:
        lr_adj = 1.
    elif epoch < 20:
        lr_adj = 5e-1
    elif epoch < 30:
        lr_adj = 1e-1
    elif epoch < 40:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(exp=args.save, epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val, name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def w_parse(weights):
    gene = []
    n = 2
    start = 0
    for i in range(node_num):      
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):        
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
            gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
    return gene


if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
    
def train(epoch, train_sampler, train_queue, valid_queue, model, criterion, optimizer):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    if torch.cuda.current_device() == 0:
        for param_group in optimizer.param_groups:
            tmp_lr = param_group['lr']
        logging.info('epoch %d lr %e', epoch, tmp_lr)
        
    with tqdm(total=len(train_queue),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for step, (data, target) in enumerate(train_queue):
            adjust_learning_rate(epoch, step)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if step % 20 == 0:
                model.sample_new_architecture()



            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = criterion(output, target)
            train_loss.update(loss)
            # Average gradients among sub-batches
            loss.backward()

            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)
    if torch.cuda.current_device() == 0:
        logging.info("epoch: " + str(epoch) + " train_acc: " + str(100. * train_accuracy.avg.item()))


def train_arch(epoch, train_queue, valid_queue, model, architect):    
    model.eval()
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(valid_queue),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        
        for step, (data, target) in enumerate(valid_queue):
            if step == 0 and torch.cuda.current_device() == 0:
                for e in model.edge_alphas_normal:
                    logging.info(F.softmax(e, dim=-1))
                for e in model.edge_alphas_reduce:
                    logging.info(F.softmax(e, dim=-1))
                logging.info(F.softmax(model.alphas_normal, dim=-1))
                logging.info(F.softmax(model.alphas_reduce, dim=-1))
                logging.info(model.baseline)

                
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            arch_optimizer.zero_grad()

            model.sample_new_architecture()

            output = model(data)
            reward = accuracy(output, target)
            val_accuracy.update(reward)
            
            sample_log_probs = model.sample_log_probs
            model.baseline = model.baseline - (1 - model.bl_dec) * (model.baseline - reward)
            loss = sample_log_probs * (reward - model.baseline)
            loss.backward()

            arch_optimizer.step()
            t.set_postfix({'accuracy': 100. * val_accuracy.avg.item()})
            t.update(1)
    if torch.cuda.current_device() == 0:
        logging.info("epoch: " + str(epoch) + " val_acc: " + str(100. * val_accuracy.avg.item()))


def best_arch_search(valid_queue, model):
    model.eval()    
    result_df = pd.DataFrame(columns = ['Normal', 'Reduce', 'Val_acc'])
    tmp_cnt = 0
    
    while(tmp_cnt < search_arch_num):  
        for step, (input_search, target_search) in enumerate(valid_queue):
            if(tmp_cnt >= search_arch_num):
                    break
            if step >= (len(valid_dataset) // args.val_batch_size):
                break

            if args.cuda:
                input_search, target_search = input_search.cuda(), target_search.cuda()

            if step % 10 == 0:
                model.sample_new_architecture()
                arch_accuracy = Metric('arch_accuray')
            output = model(input_search)
            prec1 = accuracy(output, target_search)
            arch_accuracy.update(prec1)

            if step % 10 == 0 and step != 0:
                gene_normal = w_parse(model.sampled_weight_normal.data.cpu().numpy())
                gene_reduce = w_parse(model.sampled_weight_reduce.data.cpu().numpy())
                temp_df = pd.DataFrame([[gene_normal, gene_reduce, arch_accuracy.avg.item()*100.]], columns = ['Normal', 'Reduce', 'Val_acc'])
                result_df = result_df.append(temp_df, ignore_index=True)
                tmp_cnt += 1
            
    result_df = result_df.sort_values(by='Val_acc', ascending=False)
    result_df.to_csv('search_result.csv')

    

hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(exp=args.save, epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
init_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.4,
                                hue=0.2),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))

lengths = [int(len(init_dataset)*args.train_portion), int(len(init_dataset)*(1-args.train_portion))]
if len(init_dataset) != sum(lengths):
    tmp_sum = len(init_dataset) - sum(lengths)
    lengths[1] += tmp_sum
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(init_dataset, lengths)
    

# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_queue = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler=train_sampler, **kwargs)

val_sampler = torch.utils.data.distributed.DistributedSampler(
    valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())
valid_queue = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.val_batch_size,
    sampler=val_sampler, **kwargs)
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

model = Network(args.init_channels, CLASSES, args.layers, criterion)

if args.cuda:
    # Move model to GPU.
    model.cuda()



# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.base_lr * hvd.size(),
                      momentum=args.momentum, weight_decay=args.wd)#, nesterov=True)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

    
arch_optimizer = torch.optim.Adam(model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)



# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(exp=args.save, epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(args.epochs), eta_min=args.learning_rate_min)

architect = Architect(model, args)    


# model_path = "./search-EXP-final/weights.pt"   
# model.load_state_dict(torch.load(model_path))


start_time = time.time()
if hvd.rank() == 0:
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model)) 
for epoch in range(resume_from_epoch, args.epochs):

    
    train(epoch, train_sampler, train_queue, valid_queue, model, criterion, optimizer)
    train_arch(epoch, train_queue, valid_queue, model, architect)

    save_checkpoint(epoch)

best_arch_search(valid_queue, model)   

print("time: " + str(time.time() - start_time))


    
