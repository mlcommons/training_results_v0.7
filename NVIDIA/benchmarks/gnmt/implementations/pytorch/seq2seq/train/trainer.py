import logging
import os
import time
from itertools import cycle

import numpy as np
import torch
import torch.optim
import torch.utils.data
from apex.parallel import DistributedDataParallel as DDP
from apex.contrib.optimizers import FusedAdam
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex import amp
from mlperf_logging.mllog import constants

from seq2seq.train.fp_optimizers import Fp16Optimizer
from seq2seq.train.fp_optimizers import DwuFp16Optimizer
from seq2seq.train.fp_optimizers import Fp32Optimizer
from seq2seq.utils import AverageMeter
from seq2seq.utils import log_event
from seq2seq.utils import sync_workers
from seq2seq.utils import get_world_size


class Seq2SeqTrainer:
    """
    Seq2SeqTrainer
    """
    def __init__(self,
                 model,
                 criterion,
                 opt_config,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=float('inf'),
                 batch_first=False,
                 save_info={},
                 save_path='.',
                 train_iterations=0,
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 math='fp32',
                 loss_scaling={},
                 cuda=True,
                 distributed=False,
                 distributed_overlap_allreduce=False,
                 distributed_overlap_num_allreduce_streams=1,
                 distributed_overlap_allreduce_messagesize=1e7,
                 distributed_overlap_allreduce_communicators=None,
                 intra_epoch_eval=0,
                 prealloc_mode='always',
                 iter_size=1,
                 verbose=False,
                 args=None):
        """
        Constructor for the Seq2SeqTrainer.

        :param model: model to train
        :param criterion: criterion (loss function)
        :param opt_config: dictionary with options for the optimizer
        :param print_freq: prints short summary every 'print_freq' iterations
        :param save_freq: saves checkpoint every 'save_freq' iterations
        :param grad_clip: coefficient for gradient clipping
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param save_info: dict with additional state stored in each checkpoint
        :param save_path: path to the directiory for checkpoints
        :param train_iterations: total number of training iterations to execute
        :param checkpoint_filename: name of files with checkpoints
        :param keep_checkpoints: max number of checkpoints to keep
        :param math: arithmetic type
        :param loss_scaling: options for dynamic loss scaling
        :param cuda: if True use cuda, if False train on cpu
        :param distributed: if True run distributed training
        :param intra_epoch_eval: number of additional eval runs within each
            training epoch
        :param prealloc_mode: controls preallocation,
            choices=['off', 'once', 'always']
        :param iter_size: number of iterations between weight updates
        :param verbose: enables verbose logging
        """
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epoch = 0
        self.save_info = save_info
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.cuda = cuda
        self.distributed = distributed
        self.print_freq = print_freq
        self.batch_first = batch_first
        self.verbose = verbose
        self.loss = None
        self.translator = None
        self.scheduler = None
        self.intra_epoch_eval = intra_epoch_eval
        self.iter_size = iter_size
        self.prealloc_mode = prealloc_mode
        self.preallocated = False
        
        # Assume multi-tensor apply if with APEX DDP
        self.args = args
        self.use_mt = (distributed  and iter_size == 1 and \
            opt_config['optimizer'] == 'FusedAdam')

        # Use APEX gradient average if gradient accumulation option enabled
        self.retain_allreduce_buffers = True if iter_size == 1 else False
        self.gradient_average = False if iter_size == 1 else True

        if cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        params = self.model.parameters()
        if math == 'fp16':
            self.model = self.model.half()
            if distributed and self.args.distributed_weight_update != 2:
                self.model = DDP(self.model,
                                 message_size=distributed_overlap_allreduce_messagesize,
                                 delay_allreduce=(not distributed_overlap_allreduce),
                                 num_allreduce_streams=distributed_overlap_num_allreduce_streams,
                                 allreduce_communicators=distributed_overlap_allreduce_communicators,
                                 retain_allreduce_buffers=self.retain_allreduce_buffers,
                                 gradient_average=self.gradient_average)

            if self.args.distributed_weight_update == 2:
                # gradient clipping maintained by DistributedFusedAdam
                self.fp_optimizer = DwuFp16Optimizer(
                    self.model,
                    loss_scale=loss_scaling['init_scale'],
                    dls_upscale_interval=loss_scaling['upscale_interval']
                    )
                params = list(self.model.parameters())
            else:
                self.fp_optimizer = Fp16Optimizer(
                    self.model, grad_clip,
                    use_mt=self.use_mt,
                    loss_scale=loss_scaling['init_scale'],
                    dls_upscale_interval=loss_scaling['upscale_interval']
                    )
                params = self.fp_optimizer.fp32_params if isinstance(self.fp_optimizer.fp32_params, list) \
                    else [self.fp_optimizer.fp32_params]
        elif math == 'fp32':
            if distributed:
                self.model = DDP(self.model,
                                 message_size=distributed_overlap_allreduce_messagesize,
                                 delay_allreduce=(not distributed_overlap_allreduce))
            self.fp_optimizer = Fp32Optimizer(self.model, grad_clip)
            # params = self.model.parameters()

        opt_name = opt_config.pop('optimizer')
        if opt_name == 'FusedAdam':
            if math == 'fp16' or math == 'fp32':
                if self.args.distributed_weight_update == 2:
                    dwu_args = self.distributed_weight_update_config
                    self.optimizer = DistributedFusedAdam(params, max_grad_norm=grad_clip,
                                                          **dwu_args, **opt_config)
                    self.optimizer.set_global_scale(1.0) # used for grad norm clipping in step function
                else:
                    # Maintain grad norm and scaling by ourselves
                    self.optimizer = FusedAdam(params, use_mt=self.use_mt, **opt_config)
            else:
                self.optimizer = FusedAdam(params, use_mt=self.use_mt, max_grad_norm=grad_clip,
                                           amp_scale_adjustment=get_world_size(), **opt_config)
        else:
            self.optimizer = torch.optim.__dict__[opt_name](params,
                                                            **opt_config)
        logging.info(f'Using optimizer: {self.optimizer}')

        log_event(key=constants.OPT_NAME,
                  value=constants.ADAM, sync=False)
        log_event(key=constants.OPT_BASE_LR,
                  value=opt_config['lr'], sync=False)
        log_event(key=constants.OPT_ADAM_BETA_1,
                  value=self.optimizer.defaults['betas'][0], sync=False)
        log_event(key=constants.OPT_ADAM_BETA_2,
                  value=self.optimizer.defaults['betas'][1], sync=False)
        log_event(key=constants.OPT_ADAM_EPSILON,
                  value=self.optimizer.defaults['eps'], sync=False)

    @property
    def distributed_weight_update_config(self):
        """
        Return a kwarg dictionary that provides arguments for the distributed
        weight update feature.
        """
        return {
            'dwu_group_size': self.args.dwu_group_size,
            'dwu_num_blocks': self.args.dwu_num_blocks,
            'dwu_num_chunks': self.args.dwu_num_chunks,
            'dwu_num_rs_pg': self.args.dwu_num_rs_pg,
            'dwu_num_ar_pg': self.args.dwu_num_ar_pg,
            'dwu_num_ag_pg': self.args.dwu_num_ag_pg,
            'overlap_reductions': self.args.dwu_overlap_reductions,
            'full_pipeline': self.args.dwu_full_pipeline,
            'compute_L2_grad_norm': self.args.dwu_grad_norm,
            'e5m2_allgather': self.args.dwu_e5m2_allgather,
            'predivide': False,
            'flat_mt': True,
        }

    def iterate(self, src, tgt, update=True, training=True):
        """
        Performs one iteration of the training/validation.

        :param src: batch of examples from the source language
        :param tgt: batch of examples from the target language
        :param update: if True: optimizer does update of the weights
        :param training: if True: executes optimizer
        """
        src, src_length = src
        tgt, tgt_length = tgt
        src_length = torch.LongTensor(src_length)
        tgt_length = torch.LongTensor(tgt_length)

        num_toks = {}
        num_toks['tgt'] = int(sum(tgt_length - 1))
        num_toks['src'] = int(sum(src_length))

        if self.cuda:
            src = src.cuda(non_blocking=True)
            tgt = tgt.cuda(non_blocking=True)

        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)
        else:
            output = self.model(src, src_length, tgt[:-1])
            tgt_labels = tgt[1:]
            T, B = output.size(0), output.size(1)

        loss = self.criterion(output.view(T * B, -1),
                              tgt_labels.contiguous().view(-1))

        loss_per_batch = torch.empty((1), dtype=torch.float, device='cpu',
            requires_grad=False, pin_memory=True)
        loss_per_batch.copy_(loss, non_blocking=True)
        loss /= (B * self.iter_size)

        if training:
            self.fp_optimizer.step(loss, self.optimizer, self.scheduler,
                                   update)

        loss_per_batch = loss_per_batch.item()
        loss_per_token = loss_per_batch / num_toks['tgt']
        loss_per_sentence = loss_per_batch / B

        return loss_per_token, loss_per_sentence, num_toks

    def feed_data(self, data_loader, training=True):
        """
        Runs training or validation on batches from data_loader.

        :param data_loader: data loader
        :param training: if True runs training else runs validation
        """
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_eval+2)[1:-1]
            iters_with_update = len(data_loader) // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)

        batch_time = AverageMeter(skip_first=False)
        data_time = AverageMeter(skip_first=False)
        losses_per_token = AverageMeter(skip_first=False)
        losses_per_sentence = AverageMeter(skip_first=False)

        tot_tok_time = AverageMeter(skip_first=False)
        src_tok_time = AverageMeter(skip_first=False)
        tgt_tok_time = AverageMeter(skip_first=False)

        batch_size = data_loader.batch_size

        end = time.time()
        for i, (src, tgt) in enumerate(data_loader):
            self.save_counter += 1
            # measure data loading time
            data_time.update(time.time() - end)

            update = False
            if i % self.iter_size == self.iter_size - 1:
                update = True

            # do a train/evaluate iteration
            stats = self.iterate(src, tgt, update, training=training)
            loss_per_token, loss_per_sentence, num_toks = stats

            # measure accuracy and record loss
            losses_per_token.update(loss_per_token, num_toks['tgt'])
            losses_per_sentence.update(loss_per_sentence, batch_size)

            # measure elapsed time
            elapsed = time.time() - end
            batch_time.update(elapsed)
            src_tok_time.update(num_toks['src'] / elapsed)
            tgt_tok_time.update(num_toks['tgt'] / elapsed)
            tot_num_toks = num_toks['tgt'] + num_toks['src']
            tot_tok_time.update(tot_num_toks / elapsed)
            self.loss = losses_per_token.avg

            if training and i in eval_iters:
                assert self.translator is not None
                test_bleu, _ = self.translator.run(calc_bleu=True,
                                                   epoch=self.epoch,
                                                   iteration=i)

                log = []
                log += [f'TRAIN [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'BLEU: {test_bleu:.2f}']
                log = '\t'.join(log)
                logging.info(log)

                self.model.train()
                self.preallocate(data_loader.batch_size,
                                 data_loader.dataset.max_len, training=True)

            if i % self.print_freq == 0:
                phase = 'TRAIN' if training else 'VALIDATION'
                log = []
                log += [f'{phase} [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                log += [f'Data {data_time.val:.2e} ({data_time.avg:.2e})']
                log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
                if self.verbose:
                    log += [f'Src tok/s {src_tok_time.val:.0f} ({src_tok_time.avg:.0f})']
                    log += [f'Tgt tok/s {tgt_tok_time.val:.0f} ({tgt_tok_time.avg:.0f})']
                    log += [f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})']
                log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                if training:
                    lr = self.optimizer.param_groups[0]['lr']
                    log += [f'LR {lr:.3e}']
                log = '\t'.join(log)
                logging.info(log)

            save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
            if training and save_chkpt:
                self.save_counter = 0
                self.save_info['iteration'] = i
                identifier = next(self.checkpoint_counter, -1)
                if identifier != -1:
                    with sync_workers() as rank:
                        if rank == 0:
                            self.save(identifier=identifier)

            end = time.time()

        tot_tok_time.reduce('sum')
        losses_per_token.reduce('mean')

        return losses_per_token.avg, tot_tok_time.avg

    def preallocate(self, batch_size, max_length, training):
        """
        Generates maximum sequence length batch and runs forward and backward
        pass without updating model parameters.

        :param batch_size: batch size for preallocation
        :param max_length: max sequence length for preallocation
        :param training: if True preallocates memory for backward pass
        """
        if self.prealloc_mode == 'always' or (self.prealloc_mode == 'once' and
                                              not self.preallocated):
            logging.info('Executing preallocation')
            torch.cuda.empty_cache()

            src_length = [max_length] * batch_size
            tgt_length = [max_length] * batch_size

            if self.batch_first:
                shape = (batch_size, max_length)
            else:
                shape = (max_length, batch_size)

            src = torch.full(shape, 4, dtype=torch.int64)
            tgt = torch.full(shape, 4, dtype=torch.int64)
            src = src, src_length
            tgt = tgt, tgt_length
            self.iterate(src, tgt, update=False, training=training)
            self.model.zero_grad()
            self.preallocated = True

    def optimize(self, data_loader):
        """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(True)
        self.model.train()
        self.preallocate(data_loader.batch_size, data_loader.dataset.max_len,
                         training=True)

        output = self.feed_data(data_loader, training=True)

        self.model.zero_grad()
        return output

    def evaluate(self, data_loader):
        """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        self.preallocate(data_loader.batch_size, data_loader.dataset.max_len,
                         training=False)

        output = self.feed_data(data_loader, training=False)

        self.model.zero_grad()
        return output

    def load(self, filename):
        """
        Loads checkpoint from filename.

        :param filename: path to the checkpoint file
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.fp_optimizer.initialize_model(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            assert self.scheduler is not None
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            logging.info(f'Loaded checkpoint {filename} (epoch {self.epoch})')
        else:
            logging.error(f'Invalid checkpoint: {filename}')

    def save(self, identifier=None, is_best=False, save_all=False):
        """
        Stores checkpoint to a file.

        :param identifier: identifier for periodic checkpoint
        :param is_best: if True stores checkpoint to 'model_best.pth'
        :param save_all: if True stores checkpoint after completed training
            epoch
        """

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_path, filename)
            logging.info(f'Saving model to {filename}')
            torch.save(state, filename)

        if self.distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        assert self.scheduler is not None
        state = {
            'epoch': self.epoch,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': getattr(self, 'loss', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))

        if identifier is not None:
            filename = self.checkpoint_filename % identifier
            write_checkpoint(state, filename)

        if is_best:
            filename = 'model_best.pth'
            write_checkpoint(state, filename)

        if save_all:
            filename = f'checkpoint_epoch_{self.epoch:03d}.pth'
            write_checkpoint(state, filename)
