import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from architect import Architect
import pandas as pd

import gc

import data
import model_search as model

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='./',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=150,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=5,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', default=True, action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', default=False, action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3.5e-4,
                    help='learning rate for the architecture encoding alpha')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)

corpus = data.Corpus(args.data)

eval_batch_size = int(0.5*args.batch_size)
test_batch_size = 1
search_arch_num = 100
arch_opt_step = 1
saved_path = './search-EXP-20190112-065938'

train_data = batchify(corpus.train, args.batch_size, args)
search_data = batchify(corpus.valid, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


ntokens = len(corpus.dictionary)
if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = model.RNNModelSearch(ntokens, args.emsize, args.nhid, args.nhidlast, 
                       args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute)

#model = torch.load(os.path.join(saved_path, 'model.pt'))
    
size = 0
for p in model.parameters():
    size += p.nelement()
logging.info('param size: {}'.format(size))
logging.info('initial genotype:')
model.sample_new_architecture()
print (model.sampled_weight)
logging.info(model.genotype())

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model
architect = Architect(parallel_model, args)

total_params = sum(x.data.nelement() for x in model.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model total parameters: {}'.format(total_params))


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]    
    batch, i = 0, 0
    model.train()
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
                
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)            
            
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            parallel_model.sample_new_architecture() 
            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
                loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:            
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| dag_epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len

def train_arch():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)    
    hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    ep_loss = 0    
    model.eval()
    while i < search_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        data_valid, targets_valid = get_batch(search_data, i, args)        

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:            
            cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)                        
            
            hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])
            
            parallel_model.sample_new_architecture() 
            if i == 0:
                for e in model.edge_weights:
                    print(F.softmax(e, dim=-1))
      
                print(F.softmax(model.weights, dim=-1))      
                print(model.baseline)
            
            if (batch+1) % arch_opt_step == 0:
                is_opt_step = True
            else:
                is_opt_step = False
                
            if i == 0:
                architect.optimizer.zero_grad()
                
            hidden_valid[s_id], raw_loss = architect.step(                    
                    hidden_valid[s_id], cur_data_valid, cur_targets_valid, is_opt_step)
            raw_loss, hidden_valid[s_id] = model._loss(hidden_valid[s_id], cur_data_valid, cur_targets_valid)
            raw_loss = raw_loss.detach()

            loss = raw_loss
            
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size   
            ep_loss += raw_loss * len(cur_data_valid)

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # total_loss += raw_loss.data        
        if batch % args.log_interval == 0 and batch > 0:
            logging.info(parallel_model.genotype())
            print(F.softmax(parallel_model.weights, dim=-1))
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| arch_epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(search_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len
    
    #Optimizer step for residual of valid queue
    if not is_opt_step:
        architect.optimizer.step() 
    
    return ep_loss.item() / len(search_data)

def best_arch_search():
    model.eval()    
    result_df = pd.DataFrame(columns = ['Genotype', 'Val_reward'])
    ntokens = len(corpus.dictionary)
    i = 0
    hidden = model.init_hidden(eval_batch_size)
    for m in range(search_arch_num):        
        parallel_model.sample_new_architecture()
        
        data, targets = get_batch(val_data, i, args)                
        targets = targets.view(-1)

        hidden = repackage_hidden(hidden)
        #log_prob, hidden = parallel_model(data, hidden)
        #loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data        
        loss, hidden = parallel_model._loss(hidden, data, targets)        

        reward = architect.reward_c / torch.exp(loss)

        gene = parallel_model.genotype()      
        temp_df = pd.DataFrame([[gene, reward.item()]], columns = ['Genotype', 'Val_reward'])
        result_df = result_df.append(temp_df, ignore_index=True)
        
        i += args.bptt
        if i >= search_data.size(0) - 2:
            i = 0
        
    result_df = result_df.sort_values(by='Val_reward', ascending=False)
    result_df.to_csv('search_result.csv')

# Loop over epochs.
lr = args.lr

if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss = train_arch()

    #val_loss = evaluate(val_data, eval_batch_size)
    logging.info('-' * 89)
    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    logging.info('-' * 89)
    
    save_checkpoint(model, optimizer, epoch, args.save)
    logging.info('Saving Normal!')
    
best_arch_search()