# Lint as: python3
"""LAMB Optimizer that merges small variables."""

from flax.optim import lamb
import jax
from jax import lax
import jax.numpy as jnp
from jax.util import partial


class BertLAMB(lamb.LAMB):
  """Layerwise adaptive moments for batch (LAMB) optimizer.

  See https://arxiv.org/abs/1904.00962
  """

  def __init__(self,
               learning_rate=None,
               beta1=0.9,
               beta2=0.999,
               weight_decay=0,
               eps=1e-6,
               num_layers=24):
    """Constructor for the LAMB optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      beta1: the coefficient used for the moving average of the gradient
        (default: 0.9).
      beta2: the coefficient used for the moving average of the squared gradient
        (default: 0.999).
      weight_decay: weight decay coefficient to apply
      eps: epsilon used for Adam update computation (default: 1e-6).
      num_layers: Number of layers in Bert model.
    """
    self._num_layers = num_layers
    super().__init__(learning_rate, beta1, beta2, weight_decay, eps)

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    beta1 = hyper_params.beta1
    beta2 = hyper_params.beta2
    weight_decay = hyper_params.weight_decay
    learning_rate = hyper_params.learning_rate

    grad_sq = lax.square(grad)
    grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
    grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

    t = step + 1.
    grad_ema_corr = grad_ema / (1. - beta1**t)
    grad_sq_ema_corr = grad_sq_ema / (1. - beta2**t)

    update = grad_ema_corr / (jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps)

    if weight_decay != 0.0:
      update += weight_decay * param

    if len(param.shape) > 1 and param.shape[0] == self._num_layers:
      norm_fn = partial(jnp.linalg.norm, keepdims=True)
      param_norm = jax.vmap(norm_fn)(param)
      update_norm = jax.vmap(norm_fn)(update)
    else:
      param_norm = jnp.linalg.norm(param)
      update_norm = jnp.linalg.norm(update)

    trust_ratio = jnp.where(
        param_norm > 0.,
        jnp.where(update_norm > 0, param_norm / update_norm, 1.0), 1.0)
    new_param = param - trust_ratio * learning_rate * update
    new_state = lamb._LAMBParamState(grad_ema, grad_sq_ema)  # pylint: disable=protected-access

    return new_param, new_state
