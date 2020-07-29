import torch
from apex.multi_tensor_apply import multi_tensor_applier

class FP16_Optimizer(object):
    """
    :class:`FP16_Optimizer` A cutdown version of apex.fp16_utils.FP16_Optimizer.
    Designed only to wrap FusedSGD.
    """

    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True):

        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        # create list for params in fp16
        self.fp16_params = []
        # maintain master weights for params in fp16
        self.fp32_from_fp16_params = []
        # create list for params in fp32
        self.fp32_params = []

        # iterate over param_groups
        for param_group in self.optimizer.param_groups:
            fp16_params = []
            fp32_from_fp16_params = []
            fp32_params = []
            # separate fp16/32 params into 2 groups
            for p in param_group['params']:
                if p.dtype == torch.float16: # fp16
                    fp16_params.append(p)
                    fp32_from_fp16_params.append(p.clone().float().detach())
                if p.dtype == torch.float32: # fp32
                    fp32_params.append(p)
            self.fp16_params.append(fp16_params)
            self.fp32_from_fp16_params.append(fp32_from_fp16_params)
            self.fp32_params.append(fp32_params) 

        if multi_tensor_applier.available:
            import amp_C
            self.overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm
        else:
            raise RuntimeError('FP16_Optimizer requires cuda extensions')

        if dynamic_loss_scale:
            if dynamic_loss_args is not None:
                raise SystemError("Do not support dynamic loss scale args for now.")
            self.dynamic_loss_scale = True
            self.cur_scale = 2**16
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2
            self.scale_window = 1000
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # for fp16 groups
        for group in self.fp16_params:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
        # for fp32 groups
        for group in self.fp32_params:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        fp16_grads = []
        fp32_grads = []
        skip = False

	# for fp16 groups
        for group in self.fp16_params:
            fp16_grad = []
            for i, p in enumerate(group):
                fp16_grad.append(p.grad)
            fp16_grads.append(fp16_grad)

        # for fp32 groups
        for group in self.fp32_params:
            fp32_grad = []
            for i, p in enumerate(group):
                fp32_grad.append(p.grad)
            fp32_grads.append(fp32_grad)       

        # nan check
        self.overflow_buf.zero_()
        for fp16_grad in fp16_grads:
            if len(fp16_grad) > 0:
                norm, norm_per_tensor = multi_tensor_applier(self.multi_tensor_l2norm,
                                                             self.overflow_buf,
                                                             [fp16_grad], True)
                if self.overflow_buf.item() != 0:
                    skip = True
        for fp32_grad in fp32_grads:
            if len(fp32_grad) > 0:
                norm, norm_per_tensor = multi_tensor_applier(self.multi_tensor_l2norm,
                                                             self.overflow_buf,
                                                             [fp32_grad], True)
                if self.overflow_buf.item() != 0:
                    skip = True

        if skip:
            self._update_scale(skip)
            return
 
        dict_fp16 = {'params': self.fp16_params, 'master': self.fp32_from_fp16_params, 'grads': fp16_grads}
        dict_fp32 = {'params': self.fp32_params, 'grads': fp32_grads}
          
        self.optimizer.step(dict_fp16=dict_fp16, dict_fp32=dict_fp32, scale=self.cur_scale)

        self._update_scale(False)
        return

    def backward(self, loss):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        scaled_loss = (loss.float()) * self.cur_scale
        scaled_loss.backward()

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            if skip:
                if self.verbose:
                    print("\nGrad overflow on iteration ", self.cur_iter)
                    print("Using dynamic loss scale of ", self.cur_scale)
                self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
                self.last_overflow_iter = self.cur_iter
            else:
                if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                    self.cur_scale *= self.scale_factor
        else:
            if skip:
                print("\nGrad overflow on iteration ", self.cur_iter)
                print("Using static loss scale of ", self.cur_scale)
        self.cur_iter +=1
        return

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp16_params'] = self.fp16_params
        state_dict['fp32_from_fp16_params'] = self.fp32_from_fp16_params
        state_dict['fp32_params'] = self.fp32_params
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        self.fp16_params = state_dict['fp16_params'] 
        self.fp32_from_fp16_params = state_dict['fp32_from_fp16_params']
        self.fp32_params = state_dict['fp32_params'] 
