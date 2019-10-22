from __future__ import print_function
import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkImageNet as Network
import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import tensorboardX
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='EINAS on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--arch', type=str, default='EINAS', help='which architecture to use')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')

parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')

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

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=200,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=50,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=250,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00003,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

args = parser.parse_args()
if torch.cuda.current_device() == 0:
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(args.save) and torch.cuda.current_device() == 0:
    os.mkdir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'a_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh) 


args.cuda = not args.no_cuda and torch.cuda.is_available()

CLASSES = 1000
 

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


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

kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)


# Set up standard ResNet-50 model.
genotype = eval("genotypes.%s" % args.arch)
model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
# model = models.resnet50()

if args.cuda:
    # Move model to GPU.
    model.cuda()

    
#model load
# model_path = "./model_imagenet_20190112/checkpoint-60.pth"
# model.load_state_dict(torch.load(model_path))


criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
criterion_smooth = criterion_smooth.cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.base_lr * hvd.size(),
                      momentum=args.momentum, weight_decay=args.wd)


# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

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

def train(epoch, criterion):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    if torch.cuda.current_device() == 0:
        for param_group in optimizer.param_groups:
            tmp_lr = param_group['lr']
        logging.info('epoch %d lr %e', epoch, tmp_lr/hvd.size())
        
    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            if args.auxiliary:
                loss_aux = criterion(output, target)
                loss += args.auxiliary_weight*loss_aux

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss.update(loss)
            train_accuracy.update(accuracy(output, target))
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)


    if torch.cuda.current_device() == 0:
        logging.info("epoch: " + str(epoch) + " train_acc: " + str(100. * train_accuracy.avg.item()))


def validate(epoch, criterion):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    val_top5 = Metric('val_top5')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output, _ = model(data)

                val_loss.update(criterion(output, target))
                top1, top5 = utils.accuracy(output, target, topk=(1, 5))
                val_accuracy.update(top1)
                val_top5.update(top5)
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if torch.cuda.current_device() == 0:
        logging.info("epoch: " + str(epoch) + " val_top1: " + str(val_accuracy.avg.item()) + " val_top5: " + str(val_top5.avg.item()))




# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
#     else:
#         lr_adj = args.gamma ** epoch
    elif epoch < 50:
        lr_adj = 1.
    elif epoch < 100:
        lr_adj = 1e-1
    elif epoch < 150:
        lr_adj = 1e-2
    elif epoch < 200:
        lr_adj = 1e-3
    else:
        lr_adj = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * lr_adj


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
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

if hvd.rank() == 0:
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model)) 

for epoch in range(resume_from_epoch, args.epochs):
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train(epoch, criterion_smooth)
    validate(epoch, criterion)
    save_checkpoint(epoch)
