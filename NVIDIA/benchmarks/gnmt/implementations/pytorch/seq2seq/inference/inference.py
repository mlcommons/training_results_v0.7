import contextlib
import logging
import os
import subprocess
import time

import torch
import torch.distributed as dist

import seq2seq.data.config as config
from seq2seq.inference.beam_search import SequenceGenerator
from seq2seq.utils import AverageMeter
from seq2seq.utils import barrier
from seq2seq.utils import get_rank
from seq2seq.utils import get_world_size
import seq2seq.inference.bleu


def gather_predictions(preds):
    world_size = get_world_size()
    if world_size > 1:
        all_preds = preds.new(world_size * preds.size(0), preds.size(1))
        all_preds_list = all_preds.chunk(world_size, dim=0)
        dist.all_gather(all_preds_list, preds)
        preds = all_preds
    return preds


class Translator:
    """
    Translator runs validation on test dataset, executes inference, optionally
    computes BLEU score using sacrebleu.
    """
    def __init__(self,
                 model,
                 tokenizer,
                 loader,
                 beam_size=5,
                 len_norm_factor=0.6,
                 len_norm_const=5.0,
                 cov_penalty_factor=0.1,
                 max_seq_len=50,
                 cuda=False,
                 print_freq=1,
                 dataset_dir=None,
                 save_path=None,
                 target_bleu=None):

        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.insert_target_start = [config.BOS]
        self.insert_src_start = [config.BOS]
        self.insert_src_end = [config.EOS]
        self.batch_first = model.batch_first
        self.cuda = cuda
        self.beam_size = beam_size
        self.print_freq = print_freq
        self.dataset_dir = dataset_dir
        self.target_bleu = target_bleu
        self.save_path = save_path

        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            max_seq_len=max_seq_len,
            cuda=cuda,
            len_norm_factor=len_norm_factor,
            len_norm_const=len_norm_const,
            cov_penalty_factor=cov_penalty_factor)

    def run(self, calc_bleu=True, epoch=None, iteration=None, summary=False,
            reference_path=None):
        """
        Runs translation on test dataset.

        :param calc_bleu: if True compares results with reference and computes
            BLEU score
        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        :param summary: if True prints summary
        :param reference_path: path to the file with reference translation
        """
        test_bleu = 0.
        break_training = False

        logging.info(f'Running evaluation on test set')
        self.model.eval()
        output = self.evaluate(epoch, iteration, summary)

        # detokenize (BPE)
        detok_output = []
        for idx, pred in output:
            pred = pred.tolist()
            detok = self.tokenizer.detokenize(pred)
            detok_output.append((idx, detok + '\n'))

        if calc_bleu:
            if detok_output:
                indices, output = zip(*detok_output)
            else:
                indices, output = [], []
            output = self.run_detokenizer(output)
            reference_path = os.path.join(self.dataset_dir,
                                          config.TGT_TEST_TARGET_FNAME)
            bleu = seq2seq.inference.bleu.compute_bleu(output, indices,
                                                       reference_path)
            logging.info(bleu)
            test_bleu = round(bleu.score, 2)
            if summary:
                logging.info(f'BLEU on test dataset: {test_bleu:.2f}')

            if self.target_bleu and test_bleu >= self.target_bleu:
                logging.info(f'Target accuracy reached')
                break_training = True

        logging.info(f'Finished evaluation on test set')

        return test_bleu, break_training

    def evaluate(self, epoch, iteration, summary):
        """
        Runs evaluation on test dataset.

        :param epoch: index of the current epoch
        :param iteration: index of the current iteration
        :param summary: if True prints summary
        """
        batch_time = AverageMeter(False)
        tot_tok_per_sec = AverageMeter(False)
        iterations = AverageMeter(False)
        enc_seq_len = AverageMeter(False)
        dec_seq_len = AverageMeter(False)
        stats = {}

        output = []

        for i, (src, indices) in enumerate(self.loader):
            translate_timer = time.time()
            src, src_length = src

            if self.batch_first:
                batch_size = src.shape[0]
            else:
                batch_size = src.shape[1]
            global_batch_size = batch_size * get_world_size()
            beam_size = self.beam_size

            bos = [self.insert_target_start] * (batch_size * beam_size)
            bos = torch.LongTensor(bos)
            if self.batch_first:
                bos = bos.view(-1, 1)
            else:
                bos = bos.view(1, -1)

            src_length = torch.LongTensor(src_length)
            stats['total_enc_len'] = int(src_length.sum())

            if self.cuda:
                src = src.cuda()
                bos = bos.cuda()

            with torch.no_grad():
                context = self.model.encode(src, src_length)
                if self.cuda:  src_length = src_length.cuda()
                context = [context, src_length, None]

                if beam_size == 1:
                    generator = self.generator.greedy_search
                else:
                    generator = self.generator.beam_search
                preds, lengths, counter = generator(batch_size, bos, context)

            stats['total_dec_len'] = lengths.sum().item()
            stats['iters'] = counter

            for idx, pred in zip(indices, preds):
                output.append((idx, pred))

            elapsed = time.time() - translate_timer
            batch_time.update(elapsed, batch_size)

            total_tokens = stats['total_dec_len'] + stats['total_enc_len']
            ttps = total_tokens / elapsed
            tot_tok_per_sec.update(ttps, batch_size)

            iterations.update(stats['iters'])
            enc_seq_len.update(stats['total_enc_len'] / batch_size, batch_size)
            dec_seq_len.update(stats['total_dec_len'] / batch_size, batch_size)

            if i % self.print_freq == 0:
                log = []
                log += f'TEST '
                if epoch is not None:
                    log += f'[{epoch}]'
                if iteration is not None:
                    log += f'[{iteration}]'
                log += f'[{i}/{len(self.loader)}]\t'
                log += f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                log += f'Decoder iters {iterations.val:.1f} ({iterations.avg:.1f})\t'
                log += f'Tok/s {tot_tok_per_sec.val:.0f} ({tot_tok_per_sec.avg:.0f})'
                log = ''.join(log)
                logging.info(log)

        tot_tok_per_sec.reduce('sum')
        enc_seq_len.reduce('mean')
        dec_seq_len.reduce('mean')
        batch_time.reduce('mean')
        iterations.reduce('sum')

        if summary and get_rank() == 0:
            time_per_sentence = (batch_time.avg / global_batch_size)
            log = []
            log += f'TEST SUMMARY:\n'
            log += f'Lines translated: {len(self.loader.dataset)}\t'
            log += f'Avg total tokens/s: {tot_tok_per_sec.avg:.0f}\n'
            log += f'Avg time per batch: {batch_time.avg:.3f} s\t'
            log += f'Avg time per sentence: {1000*time_per_sentence:.3f} ms\n'
            log += f'Avg encoder seq len: {enc_seq_len.avg:.2f}\t'
            log += f'Avg decoder seq len: {dec_seq_len.avg:.2f}\t'
            log += f'Total decoder iterations: {int(iterations.sum)}'
            log = ''.join(log)
            logging.info(log)

        return output

    def run_detokenizer(self, data):
        """
        Executes moses detokenizer.

        :param data: list of sentences to detokenize
        """

        data = ''.join(data)
        detok_path = os.path.join(self.dataset_dir, config.DETOKENIZER)
        cmd = f'perl {detok_path}'
        logging.info('Running moses detokenizer')
        z = subprocess.run(cmd, shell=True, input=data.encode(),
                           stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
        output = z.stdout.decode().splitlines()
        return output
