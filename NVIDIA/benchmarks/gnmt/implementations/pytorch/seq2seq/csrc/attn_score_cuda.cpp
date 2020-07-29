#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor attn_score_forward_cuda(
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn);

std::vector<at::Tensor> attn_score_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor attn_score_forward(
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn) {
    CHECK_INPUT(attn_query);
    CHECK_INPUT(attn_keys);
    CHECK_INPUT(bias);
    CHECK_INPUT(linear_attn);

    return attn_score_forward_cuda(attn_query, attn_keys, bias, linear_attn);
}

std::vector<at::Tensor> attn_score_backward(
    const at::Tensor &grad_output,
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(attn_query);
    CHECK_INPUT(attn_keys);
    CHECK_INPUT(bias);
    CHECK_INPUT(linear_attn);

    return attn_score_backward_cuda(grad_output, attn_query, attn_keys, bias, linear_attn);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attn_score_forward, "Attention score calculation forward (CUDA)");
    m.def("backward", &attn_score_backward, "Attention score calculation backward (CUDA)");
}
