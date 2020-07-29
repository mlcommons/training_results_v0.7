import logging
import math

import torch
from torch.nn.utils import clip_grad_norm_

import seq2seq.utils as utils
from apex.contrib.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
from amp_C import multi_tensor_l2norm

import apex.amp._amp_state
from apex import amp


class Fp16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
    """
    # Flattening master weight
    def initialize_flat_fp32_weight(self, model):
        logging.info('Initializing fp32 clone weights')
        self.fp16_model = model
        for p in self.fp16_model.parameters():
            p.grad = None

        nelem = 0
        for p in model.parameters():
            nelem += p.numel()
        self.fp32_params = torch.cuda.FloatTensor(nelem)
        self.fp16_params = torch.cuda.HalfTensor(nelem)

        pointer = 0
        for p in model.parameters():
            nelem = p.numel()
            self.fp32_params[pointer:pointer+nelem].copy_(p.data.view(-1))
            self.fp16_params[pointer:pointer+nelem].copy_(p.data.view(-1))
            pointer += nelem

        self.fp32_params = torch.nn.Parameter(self.fp32_params)
        self.fp32_params.grad = torch.autograd.Variable(
            self.fp32_params.data.new(*self.fp32_params.size()))
        self.fp16_params = torch.nn.Parameter(self.fp16_params)
        self.fp16_params.grad = torch.autograd.Variable(
            self.fp16_params.data.new(*self.fp16_params.size()))

    @staticmethod
    def fp16_to_fp32_flat_grad(fp32_params, fp16_model):
        pointer = 0
        for p in fp16_model.parameters():
            nelem = p.numel()
            fp32_params.grad.data[pointer:pointer+nelem].copy_(p.grad.data.view(-1))
            pointer += nelem

    @staticmethod
    def fp16_to_fp16_flat_grad(fp16_params, fp16_model):
        fp16_params.grad.data = torch.cat(
            [p.grad.data.view(-1) for p in fp16_model.parameters()])

    @staticmethod
    def fp32_to_fp16_params(fp16_model, fp32_params):
        #Copy master weights onto model weights
        pointer = 0
        for p in fp16_model.parameters():
            nelem = p.numel()
            p.data.view(-1).copy_(fp32_params.data[pointer:pointer+nelem])
            pointer += nelem


    def __init__(self, fp16_model, grad_clip=float('inf'), loss_scale=1024,
                 dls_downscale=2, dls_upscale=2, dls_upscale_interval=128,
                 use_mt=False):
        """
        Constructor for the Fp16Optimizer.

        :param fp16_model: model (previously casted to half)
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        :param loss_scale: initial loss scale
        :param dls_downscale: loss downscale factor, loss scale is divided by
            this factor when NaN/INF occurs in the gradients
        :param dls_upscale: loss upscale factor, loss scale is multiplied by
            this factor if previous dls_upscale_interval batches finished
            successfully
        :param dls_upscale_interval: interval for loss scale upscaling
        :param use_mt: with multi-tensor apply we don't need to flatten parameters
        """
        logging.info('Initializing fp16 optimizer with {}'.format(
            'multi-tenosr apply' if use_mt else 'flattening'))
        if use_mt:
            self.initialize_model(fp16_model)
        else:
            self.initialize_flat_fp32_weight(fp16_model)

        self.use_mt = use_mt
        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.grad_clip = grad_clip
        self.world_size = utils.get_world_size()

        self.dummy_overflow_buf = torch.cuda.IntTensor([0])

    def initialize_model(self, model):
        """
        Initializes internal state and build fp32 master copy of weights.

        :param model: fp16 model
        """
        logging.info('Initializing fp32 clone weights')
        self.fp16_model = model
        for p in self.fp16_model.parameters():
            p.grad = None
        self.fp32_params = [param.to(torch.float32).detach()
                            for param in model.parameters()]
        self.fp16_params = [p for p in model.parameters()]

        for param in self.fp32_params:
            param.requires_grad = True

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.
        Applies loss scaling, computes gradients in fp16, converts gradients to
        fp32, inverts scaling and applies optional gradient norm clipping.
        If gradients are finite, it applies update to fp32 master weights and
        copies updated parameters to fp16 model for the next iteration. If
        gradients are not finite, it skips the batch and adjusts scaling factor
        for the next iteration.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss *= self.loss_scale
        loss.backward()

        if not update:  return

        # Average the all-reduced gradients by world size if APEX
        # doesn't do that
        scaling_factor = self.loss_scale
        if hasattr(self.fp16_model, 'gradient_average') and \
                not self.fp16_model.gradient_average:
            scaling_factor *= self.world_size

        # APEX DDP reset the gradients to be views into allreduce_buffers
        # So downstream code should simply be able to use the .grad
        # attributes as usual
        if isinstance(optimizer, FusedAdam):
            if self.world_size != 1 and self.fp16_model.retain_allreduce_buffers:
                grads = [p.grad for p in self.fp16_params]
                norm, _ = multi_tensor_applier(
                        multi_tensor_l2norm,
                        self.dummy_overflow_buf,
                        [grads],
                        False)
                norm = norm.item() / scaling_factor
            else:
                self.fp16_to_fp16_flat_grad(self.fp16_params, self.fp16_model)
                grads = [self.fp16_params.grad]
                norm = self.fp16_params.grad.data.norm(p=2,
                    dtype=torch.float).item() / scaling_factor
        else:
            self.fp16_to_fp32_flat_grad(self.fp32_params, self.fp16_model)
            if scaling_factor != 1.0:
                self.fp32_params.grad.data /= scaling_factor

            norm = clip_grad_norm_([self.fp32_params], self.grad_clip)

        if math.isfinite(norm):
            if scheduler is not None:
                scheduler.step()

            if isinstance(optimizer, FusedAdam):
                clip_coef = self.grad_clip / (norm + 1e-6)
                clip_coef = scaling_factor / min(1, clip_coef)
                if self.use_mt:
                    optimizer.step(grads=grads, output_params=self.fp16_params, scale=clip_coef)
                else:
                    optimizer.step(grads=grads, scale=clip_coef)
            else:
                optimizer.step()

            # Unflatten params if not multi-tensor apply
            if not self.use_mt:
                self.fp32_to_fp16_params(self.fp16_model, self.fp32_params)
            self.since_last_invalid += 1
        else:
            self.loss_scale /= self.dls_downscale
            self.since_last_invalid = 0
            logging.info(f'Gradient norm: {norm}')
            logging.info(f'Skipped batch, new scale: {self.loss_scale}')

        if self.since_last_invalid >= self.dls_upscale_interval:
            self.loss_scale *= self.dls_upscale
            self.loss_scale = min(self.loss_scale, 8192.0)
            logging.info(f'Upscaling, new scale: {self.loss_scale}')
            self.since_last_invalid = 0

        for p in self.fp16_model.parameters():
            p.grad = None


class DwuFp16Optimizer:
    """
    Distributed weight update mixed precision optimizer with dynamic
    loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
    """
    def __init__(self, fp16_model, loss_scale=1024,
                 dls_downscale=2, dls_upscale=2, dls_upscale_interval=128):
        """
        Constructor for the DwuFp16Optimizer.

        :param fp16_model: model (previously casted to half)
        :param loss_scale: initial loss scale
        :param dls_downscale: loss downscale factor, loss scale is divided by
            this factor when NaN/INF occurs in the gradients
        :param dls_upscale: loss upscale factor, loss scale is multiplied by
            this factor if previous dls_upscale_interval batches finished
            successfully
        :param dls_upscale_interval: interval for loss scale upscaling
        """
        logging.info('Initializing dwu fp16 optimizer')

        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.world_size = utils.get_world_size()
        self.fp16_model = fp16_model

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.
        Applies loss scaling, computes gradients in fp16, converts gradients to
        fp32, inverts scaling and applies optional gradient norm clipping.
        If gradients are finite, it applies update to fp32 master weights and
        copies updated parameters to fp16 model for the next iteration. If
        gradients are not finite, it skips the batch and adjusts scaling factor
        for the next iteration.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        scaling_factor = self.loss_scale * self.world_size
        optimizer.set_global_scale(scaling_factor)

        loss *= self.loss_scale
        loss.backward()
        optimizer.complete_reductions()

        if not update:
            torch.cuda.synchronize()
            return

        # Gradient division by world_size is fused with FusedAdam
        norm = optimizer.L2_grad_norm / scaling_factor
        should_update = math.isfinite(norm)
        if should_update:
            if scheduler is not None:
                scheduler.step()
            optimizer.step(skip_overflow_check=True)

        if should_update:
            self.since_last_invalid += 1
        else:
            self.loss_scale /= self.dls_downscale
            self.since_last_invalid = 0
            logging.info(f'Gradient norm: {norm}')
            logging.info(f'Skipped batch, new scale: {self.loss_scale}')

        if self.since_last_invalid >= self.dls_upscale_interval:
            self.loss_scale *= self.dls_upscale
            self.loss_scale = min(self.loss_scale, 8192.0)
            logging.info(f'Upscaling, new scale: {self.loss_scale}')
            self.since_last_invalid = 0

        for p in self.fp16_model.parameters():
            p.grad = None


class Fp32Optimizer:
    """
    Standard optimizer, computes backward and applies weight update.
    """
    def __init__(self, model, grad_clip=None):
        """
        Constructor for the Fp32Optimizer

        :param model: model
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        """
        logging.info('Initializing fp32 optimizer')
        self.initialize_model(model)
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        """
        Initializes state of the model.

        :param model: model
        """
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss.backward()
        if update:
            if self.grad_clip != float('inf'):
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if scheduler is not None:
                scheduler.step()
            optimizer.step()
            self.model.zero_grad()


