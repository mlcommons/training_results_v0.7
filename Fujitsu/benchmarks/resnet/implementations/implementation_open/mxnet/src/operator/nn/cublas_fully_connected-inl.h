/*!
 * Copyright (c) 2017 by Contributors
 * \file cublas_fully_connected-inl.h
 * \brief fully connect operator and symbol with direct use of cuBLAS
*/
#ifndef MXNET_OPERATOR_NN_CUBLAS_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_NN_CUBLAS_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "./fully_connected-inl.h"
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDA && CUDA_VERSION >= 8000

/**
 * \brief This is the implementation of fully connected operator for cuBLAS.
 */
template<typename DType>
class CuBLASFullyConnectedOp {
 public:
  CuBLASFullyConnectedOp() { }

  ~CuBLASFullyConnectedOp() {  }

  void Init(const FullyConnectedParam &p,
                                  const Context& ctx) {
    using namespace mshadow;
    this->param_ = p;
    verbose_msg_logged = false;
    // If no local setting for TensorCore use policy, look to global policy.
    if (!param_.cublas_tensor_core.has_value())
      param_.cublas_tensor_core = GetEnvAllowTensorCore();
    if (!SupportsFloat16Compute(ctx.dev_id)) {
      // Only warn user if the precision was set explicitly, i.e. silently
      // fail-over to pseudo-fp16 on Maxwell or earlier GPUs.
      if (param_.cublas_algo_fwd_prec == kFloat16 || param_.cublas_algo_bwd_prec == kFloat16)
        LOG(WARNING) << "Requested FullyConnected fp16 compute precision " <<
                        "not supported by this gpu arch, using fp32 instead.";
      // Remap precision to float32 whenever float16 would have been attempted.
      if (param_.cublas_algo_fwd_prec == kFloat16 ||
          (param_.cublas_algo_fwd_prec == -1 && DataType<DType>::kFlag == kFloat16))
        param_.cublas_algo_fwd_prec = kFloat32;
      if (param_.cublas_algo_bwd_prec == kFloat16 ||
          (param_.cublas_algo_bwd_prec == -1 && DataType<DType>::kFlag == kFloat16))
        param_.cublas_algo_bwd_prec = kFloat32;
    }
    // If the user didn't specify the algos, set up the proper default.
    int default_algo = static_cast<int>(CUBLAS_GEMM_DFALT);
    #if CUDA_VERSION >= 9000
      // The cublas_tensor_core flag specifies the policy for fp16-I/O GEMMs only.
      // While TensorCore algos exist for fp32-I/O GEMMs, we do not enable it by
      // default based on this flag.
      if (DataType<DType>::kFlag == kFloat16 && param_.cublas_tensor_core.value())
        default_algo = static_cast<int>(CUBLAS_GEMM_DFALT_TENSOR_OP);
    #endif
    if (!param_.cublas_algo_fwd.has_value())
      param_.cublas_algo_fwd = default_algo;
    if (!param_.cublas_algo_bwd_data.has_value())
      param_.cublas_algo_bwd_data = default_algo;
    if (!param_.cublas_algo_bwd_weights.has_value())
      param_.cublas_algo_bwd_weights = default_algo;
  }

  void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data) {
    using mshadow::Shape2;
    using mshadow::expr::repmat;

    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, mshadow::Stream<gpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";

    // Note: data is in row-major order.  Assume a data input with batch size 'B'
    // and a number of input features 'IF', and an output with batch size 'B'
    // and output features 'OF'.  Then the shape of this operation's I/O's are:
    //
    // Data input: B x IF
    // Weights input: OF x IF
    // Output: B x OF
    //
    // Now consider an operation dot(A,B) -> C, where
    //
    // A has shape (m,k)
    // B has shape (k,n)
    // C has shape (m,n)
    //
    // Matching C's shape to our Output, we have m=B and n=OF. What remains is k=IF.
    //
    // dot ( m x k , k x n ) -> m x n
    // dot ( B x IF, IF x OF ) -> B x OF
    // dot ( Data, Weights.transpose() ) -> Output
    //

    int m = 0, k = 0, n = 0;
    GetForwardMKN(in_data[fullc::kData], in_data[fullc::kWeight], out_data[fullc::kOut],
                  param_.flatten,
                  &m, &k, &n);

    // Log algos selected.  No selection shows up as -1 or 99.  This occurs here rather than in
    // the constructor so that layer shape info can be displayed.
    if (param_.cublas_algo_verbose && !verbose_msg_logged) {
      LOG(INFO) << "FC layer algo with (batchsize, in-features, out-features) = " <<
                "(" << m << "," << k << "," << n << ")";
      LOG(INFO) << "              forward: " << param_.cublas_algo_fwd.value();
      LOG(INFO) << "     backprop-to-data: " << param_.cublas_algo_bwd_data.value();
      LOG(INFO) << "  backprop-to-weights: " << param_.cublas_algo_bwd_weights.value();
      LOG(INFO) << "";
      verbose_msg_logged = true;
    }

    mshadow::Tensor<gpu, 2, DType> data =
        in_data[fullc::kData].get_with_shape<gpu, 2, DType>(Shape2(m, k), s);
    mshadow::Tensor<gpu, 2, DType> wmat =
        in_data[fullc::kWeight].get<gpu, 2, DType>(s);
    mshadow::Tensor<gpu, 2, DType> out =
        out_data[fullc::kOut].get_with_shape<gpu, 2, DType>(Shape2(m, n), s);

    // Performs fully_connected-inl.h line:     out = dot(data, wmat.T());

    ExpandedDot(s, false, true, kWriteTo,
               data, wmat, out,
               param_.cublas_algo_fwd_prec,
               param_.cublas_algo_fwd.value());

    if (!param_.no_bias) {
      mshadow::Tensor<gpu, 1, DType> bias =
          in_data[fullc::kBias].get_with_shape<gpu, 1, DType>(Shape1(n), s);
      AddBias(bias, data, out, s);
    }
  }

  void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad) {
    using mshadow::Shape2;
    using mshadow::expr::sum_rows;

    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(s->blas_handle_ownership_, mshadow::Stream<gpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";

    // For back-prop to weights:
    //
    // Data input: B x IF
    // "Output" gradient (an input): B x OF
    // Weights gradient (an output): OF x IF
    //
    // Matching C's shape to the Weights grad, we have m=OF and n=IF. What remains is k=B.
    //
    // dot ( m x k , k x n ) -> m x n
    // dot ( OF x B, B x IF ) -> OF x IF
    // dot ( OutGradient.transpose(), Data ) -> Weights

    int m = 0, k = 0, n = 0;
    GetForwardMKN(in_data[fullc::kData], in_data[fullc::kWeight], out_grad[fullc::kOut],
                  param_.flatten,
                  &m, &k, &n);

    mshadow::Tensor<gpu, 2, DType> data =
        in_data[fullc::kData].get_with_shape<gpu, 2, DType>(Shape2(m, k), s);
    mshadow::Tensor<gpu, 2, DType> wmat =
        in_data[fullc::kWeight].get<gpu, 2, DType>(s);
    mshadow::Tensor<gpu, 2, DType> grad =
        out_grad[fullc::kOut].get_with_shape<gpu, 2, DType>(Shape2(m, n), s);

    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    mshadow::Tensor<gpu, 2, DType> gwmat = in_grad[fullc::kWeight].get<gpu, 2, DType>(s);

    // Performs fully_connected-inl.h: Assign(gwmat, req[fullc::kWeight], dot(grad.T(), data));

    ExpandedDot(s, true, false, req[fullc::kWeight],
                grad, data, gwmat,
                param_.cublas_algo_bwd_prec,
                param_.cublas_algo_bwd_weights.value());

    // gradient of bias

    if (!param_.no_bias) {
      AddBiasGrad(in_grad[fullc::kBias], grad, req[fullc::kBias], param_.num_hidden, ctx);
    }

    // gradient of data

    // "Output" gradient (an input): B x OF
    // Weights : OF x IF
    // Data gradient (an output): B x IF
    //
    // Matching C's shape to the Data gradient output, we have m=B and n=IF. What remains is k=OF.
    //
    // dot ( m x k , k x n ) -> m x n
    // dot ( B x OF, OF x IF ) -> B x IF
    // dot ( OutGradient, Weights ) -> Data Gradient

    mshadow::Tensor<gpu, 2, DType> gdata =
        in_grad[fullc::kData].get_with_shape<gpu, 2, DType>(Shape2(m, k), s);

    // Performs fully_connected-inl.h line: Assign(gdata, req[fullc::kData], dot(grad, wmat));
    ExpandedDot(s, false, false, req[fullc::kData],
                grad, wmat, gdata,
                param_.cublas_algo_bwd_prec,
                param_.cublas_algo_bwd_data.value());
  }

  /*!
   * \brief Returns whether the cublas library supports the fully-connected
   * operation described by `param`.
   */
  static bool Supports(FullyConnectedParam param,
                                  const Context& ctx) {
    // This operator uses cublasGemmEx(), which is only supported on cuda
    // compute architectures >= 5.0.  The FullyConnectedParam argument
    // is currently not considered, although this may change in the future.

    if (ComputeCapabilityMajor(ctx.dev_id) >= 5)
      return true;
    // Warn users that have specified a non-default setting for the fine-grained control.
    if (!param.cublas_off && (param.cublas_algo_fwd ||
                          param.cublas_algo_bwd_data ||
                          param.cublas_algo_bwd_weights ||
                          param.cublas_algo_fwd_prec != -1 ||
                          param.cublas_algo_bwd_prec != -1)) {
      LOG(WARNING) << "Fine-grained FC layer control not possible on this GPU architecture.";
    }
    return false;
  }

 private:
  // Return a pointer to the value '1' in the format required by `algo_precision`.
  // Used typically for the 'alpha' parameter to the GEMM kernel.
  static const void *one(int32_t algo_precision) {
    using namespace mxnet::common::cuda;
    // Default algo precision is that set by DType
    const void *answer = &CublasType<DType>::one;
    if (algo_precision != -1) {
      MSHADOW_REAL_TYPE_SWITCH(algo_precision, CType, {
        answer = &CublasType<CType>::one;
      })
    }
    return answer;
  }

  // Return a pointer to the value '0' in the format required by `algo_precision`.
  // Used typically for the 'beta' parameter to the GEMM kernel.
  static const void *zero(int32_t algo_precision) {
    using namespace mxnet::common::cuda;
    // Default algo precision is that set by DType
    const void *answer = &CublasType<DType>::zero;
    if (algo_precision != -1) {
      MSHADOW_REAL_TYPE_SWITCH(algo_precision, CType, {
        answer = &CublasType<CType>::zero;
      })
    }
    return answer;
  }

  // Returns the matrix multiply parameters m, k, and n of the forward inference operation:
  //
  // (m x k) matrix-multiply (k x n) -> (m x n)
  //
  // Similar to the code in fully_connected-inl.h, the TBlob shapes are effectively flattened if
  // they are not 2D.
  static void GetForwardMKN(const TBlob &data,
                            const TBlob &weights,
                            const TBlob &output,
                            bool flatten,
                            int *m_ptr, int *k_ptr, int *n_ptr) {
    const TShape& ishape = data.shape_;
    const TShape& wshape = weights.shape_;
    const TShape& oshape = output.shape_;

    int m = flatten ? ishape[0] : ishape.ProdShape(0, ishape.ndim()-1);
    int k = flatten ? ishape.ProdShape(1, ishape.ndim()) : ishape[ishape.ndim()-1];
    int n = wshape[0];  // Weight matrix is transposed in forward inference
    // Check consistency of input and output shapes
    int k2 = wshape.ProdShape(1, wshape.ndim());
    int m2 = flatten ? oshape[0] : oshape.ProdShape(0, oshape.ndim()-1);
    int n2 = flatten ? oshape.ProdShape(1, oshape.ndim()) : oshape[oshape.ndim()-1];

    CHECK_EQ(m, m2) << "In FullyConnected GEMM, 'data' matrix rows (" << m << ")"
                    << " must match output matrix rows (" << m2 << ")";
    CHECK_EQ(k, k2) << "In FullyConnected GEMM, 'data' matrix cols (" << k << ")"
                    << " must match 'weight' matrix rows (" << k2 << ")";
    CHECK_EQ(n, n2) << "In FullyConnected GEMM, 'data' matrix cols (" << n << ")"
                    << " must match output matrix cols (" << n2 << ")";
    *m_ptr = m;
    *k_ptr = k;
    *n_ptr = n;
  }

  // Perform the matrix multiplication (a.k.a. 'dot') on the supplied Tensors,
  // converting between the row-major specification of this routine's interface/Tensors
  // and the column-major interface of the underlying cuBLAS gemm API.
  static void ExpandedDot(mshadow::Stream<gpu> *s, bool transposeA, bool transposeB,
                          OpReqType output_request,
                          const mshadow::Tensor<gpu, 2, DType> &A,
                          const mshadow::Tensor<gpu, 2, DType> &B,
                          const mshadow::Tensor<gpu, 2, DType> &C,
                          int32_t algo_precision,
                          int algo) {
    int m = transposeA ? A.shape_[1] : A.shape_[0];
    int k = transposeA ? A.shape_[0] : A.shape_[1];
    int n = transposeB ? B.shape_[0] : B.shape_[1];
    // Check consistency of input and output shapes by grabbing n, m and k a different way.
    int k2 = transposeB ? B.shape_[1] : B.shape_[0];
    int m2 = C.shape_[0];
    int n2 = C.shape_[1];

    CHECK_EQ(m, m2) << "In FullyConnected GEMM, 'data' matrix rows (" << m << ")"
                    << " must match output matrix rows (" << m2 << ")";
    CHECK_EQ(k, k2) << "In FullyConnected GEMM, 'data' matrix cols (" << k << ")"
                    << " must match 'weight' matrix rows (" << k2 << ")";
    CHECK_EQ(n, n2) << "In FullyConnected GEMM, 'data' matrix cols (" << n << ")"
                    << " must match output matrix cols (" << n2 << ")";

    // We now juggle the arguments of the matrix multiply to account for the
    // fact that the data as generated by mxnet is in row-major order, yet the
    // BLAS gemm kernel assumes they are in column-major order.
    //
    // Let .T() represent the transpose operation below.
    //
    // If A matrix-multiply B -> C, then B.T() matrix-multiply A.T() -> C.T()
    //
    // The ramifications of this are that in order for the gemm kernel to generate
    // the correct result we need to do 2 things:
    //
    // 1. swap the input ordering (so matrix B is the first operand)
    // 2. swap the dimensions of the matrices.
    //
    // The effect of these two steps effectively swaps n and m, leaving k the same.
    // Keep in mind that any transposition that needs to be done moves with the
    // input, so the transpose arguments are swapped as well.
    // In order to operate on submatrices in a "row-major world", one needs the
    // row-stride, which is the second dimension of the two for each matrix.

    int lda = B.shape_[1];
    int ldb = A.shape_[1];
    int ldc = C.shape_[1];

    CublasGemm(s, transposeB, transposeA, n, m, k,
               output_request,
               B,
               A,
               C,
               lda, ldb, ldc,
               algo_precision,
               algo);
  }

  // A wrapper for the full-featured matrix multiply kernel cublasGemmEx() available
  // since CUDA 8 that permits data-type, compute-type and algorithm selection.
  static void CublasGemm(mshadow::Stream<gpu> *s, bool transposeA, bool transposeB,
                         int m, int n, int k,
                         OpReqType output_request,
                         const mshadow::Tensor<gpu, 2, DType> &A,
                         const mshadow::Tensor<gpu, 2, DType> &B,
                         const mshadow::Tensor<gpu, 2, DType> &C,
                         int lda, int ldb, int ldc,
                         int32_t algo_precision,
                         int algo) {
    using namespace mxnet::common::cuda;

    // The default is to have the compute type be the same as the data type,
    // unless the data type is float16, in which case float32 is chosen.
    auto compute_precision = CublasType<DType>::kCudaFlag;
    if (CublasType<DType>::kFlag == mshadow::kFloat16 &&
        algo_precision == -1) {
      algo_precision = mshadow::kFloat32;
    }
    // If the user has overriden this default, change the compute_precision enum
    // as well as the format of the alpha and beta constants.
    if (algo_precision != -1) {
      MSHADOW_REAL_TYPE_SWITCH(algo_precision, CType, {
        compute_precision = CublasType<CType>::kCudaFlag;
      })
    }
    const void *alpha = one(algo_precision);
    const void *beta = zero(algo_precision);

    switch (output_request) {
      case kNullOp:
        break;
      case kAddTo:
        // Change beta to point to 1 (the alpha value) rather than 0 to achieve summation.
        beta = alpha;
      case kWriteTo:
      case kWriteInplace: {
          auto blas_handle = mshadow::Stream<gpu>::GetBlasHandle(s);

          #if CUDA_VERSION >= 9000
            auto handle_math_mode = CUBLAS_DEFAULT_MATH;
            CUBLAS_CALL(cublasGetMathMode(blas_handle, &handle_math_mode));
            // The math mode of the handle needs to be set in sync with the math mode that
            // is baked into the algo number.  Algo numbers at or above CUBLAS_GEMM_DFALT_TENSOR
            // currently have names that include "TENSOR_OP".
            auto desired_math_mode = (algo >= CUBLAS_GEMM_DFALT_TENSOR_OP) ? CUBLAS_TENSOR_OP_MATH
                                                                           : CUBLAS_DEFAULT_MATH;
            if (handle_math_mode != desired_math_mode)
              CUBLAS_CALL(cublasSetMathMode(blas_handle, desired_math_mode));
          #endif

          auto err = CUBLAS_STATUS_SUCCESS;
          err = cublasGemmEx(blas_handle,
                             CublasTransposeOp(transposeA), CublasTransposeOp(transposeB),
                             m, n, k,
                             alpha,
                             A.dptr_, CublasType<DType>::kCudaFlag, lda,  // A operand
                             B.dptr_, CublasType<DType>::kCudaFlag, ldb,  // B operand
                             beta,
                             C.dptr_, CublasType<DType>::kCudaFlag, ldc,  // C operand
                             compute_precision,
                             static_cast<cublasGemmAlgo_t>(algo));
          CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas gemmEx fail.";

          #if CUDA_VERSION >= 9000
            if (handle_math_mode != desired_math_mode)
              CUBLAS_CALL(cublasSetMathMode(blas_handle, handle_math_mode));
          #endif
        }
        break;
      default:
        LOG(FATAL) << "not reached";
    }
  }

  FullyConnectedParam param_;
  bool verbose_msg_logged;
};  // class CuBLASFullyConnectedOp
#endif  // MXNET_USE_CUDA

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUBLAS_FULLY_CONNECTED_INL_H_
