#include "ATen/cuda/CUDAContext.h"
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/types.h>
namespace at {
namespace native {

namespace {
template <typename scalar_t>
__global__ void revert_varlen_kernel(scalar_t *in, scalar_t *out,
                                     int64_t *offsets, int feature_size, int n,
                                     scalar_t pad_value) {
  const int offset = static_cast<int>(offsets[blockIdx.x]);
  for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
    out[blockIdx.x * feature_size + i] =
        (offset >= 0) ? in[offset + i] : pad_value;
  }
}

} // namespace

void checkLongTensor(const Tensor &tensor) {
  TORCH_CHECK(tensor.dim() == 1 && tensor.device() == at::kCPU &&
                  tensor.scalar_type() == at::kLong,
              "'lengths' argument should be a 1D CPU int64 tensor");
}

at::Tensor revert_varlen_tensor(const Tensor &_input, const Tensor &_offsets) {
  auto input = _input.contiguous();
  auto output = torch::empty_like(input);
  int64_t seq_length = input.size(0);
  int64_t batch_size = input.size(1);

  assert(_offsets.dim() == 1);
  assert(_offsets.is_cuda());
  assert(_offsets.scalar_type() == at::kLong);
  TORCH_CHECK(_offsets.dim() == 1 && _offsets.is_cuda() &&
                  _offsets.scalar_type() == at::kLong,
              "'offsets' argument should be a 1D CUDA int64 tensor");
  TORCH_CHECK(_offsets.numel() == batch_size * seq_length,
              "Expected `len(offsets) = batch_size * seq_length`, but got ",
              _offsets.numel(), " (batch_size=", batch_size,
              ", seq_length=", seq_length, ")");

  int64_t feature_size = 1;
  for (int64_t dim = 2; dim < input.ndimension(); dim++) {
    feature_size *= input.size(dim);
  }

  int numThreads = 512;
  int numBlocks = batch_size * seq_length;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "revert_varlen", [&] {
        revert_varlen_kernel<<<numBlocks, numThreads, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            _offsets.data_ptr<int64_t>(), feature_size, batch_size * seq_length,
            static_cast<scalar_t>(0));
      });

  return output;
}

at::Tensor get_offsets(const Tensor &_input, const Tensor &_lengths) {
  at::native::checkLongTensor(_lengths);
  auto input = _input.contiguous();
  int64_t seq_length = input.size(0);
  int64_t batch_size = input.size(1);
  int64_t *lengths = _lengths.data_ptr<int64_t>();

  TORCH_CHECK(_lengths.size(0) == batch_size,
              "Expected `len(lengths)` to be equal to batch_size, but got ",
              _lengths.size(0), " (batch_size=", batch_size, ")");
  TORCH_CHECK(
      (lengths[batch_size - 1] > 0),
      "Length of all samples has to be greater than 0, but found an element "
      "in 'lengths' that is <= 0");

  std::vector<int64_t> offsets;
  offsets.reserve(batch_size * seq_length);
  int64_t feature_size = 1;
  for (int64_t dim = 2; dim < input.ndimension(); dim++) {
    feature_size *= input.size(dim);
  }
  for (int64_t t = 0; t < seq_length; t++) {
    for (int64_t i = 0; i < batch_size; i++) {
      if (lengths[i] > t) {
        offsets.push_back(i * feature_size +
                          (lengths[i] - t - 1) * batch_size * feature_size);
      } else {
        offsets.push_back(-1);
      }
    }
  }

  auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  auto offsets_tensor =
      at::from_blob(offsets.data(), batch_size * seq_length, at::kLong)
          .to(options, /* non_blocking */ true, /*copy*/ false);
  return offsets_tensor;
}

} // namespace native
} // namespace at
