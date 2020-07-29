/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu, Chris Olivier, Da Zheng
*/

#include "batch_norm_add_relu-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"
#include "../operator_common.h"

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

namespace mxnet {
namespace op {
namespace batchnormaddrelu {

/*! \brief Global disable of batchnormaddrelu mkl operator for unit testing */
volatile bool disable_mkl = false;

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType, typename OnData>
static inline void ForEachFast(const BNTensor3<DType> &tensor,
                               const size_t channel,
                               OnData onData) {
  const size_t num        = tensor.OuterSize();
  const size_t matrixSize = tensor.InnerSize();
  const size_t skipLength = tensor.SkipLengthToNextSameChannelData();
  const size_t startOffset = tensor.StartOffset(channel);
  DType *data = tensor.dptr_ + startOffset;

  for (size_t outer = 0; outer < num; ++outer) {
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++);
    }
    data += skipLength;
  }
}

/*! \brief Fast-foreach when you don't care about the position other than channel */
template<typename DType1, typename DType2, typename OnData>
static inline void ForEachFast(const BNTensor3<DType1> &in_data,
                               const BNTensor3<DType2> &out_data,
                               const size_t channel,
                               OnData onData) {
  const size_t num         = in_data.OuterSize();
  const size_t matrixSize  = in_data.InnerSize();
  const size_t skipLength  = in_data.SkipLengthToNextSameChannelData();
  const size_t startOffset = in_data.StartOffset(channel);

  DType1  *data = in_data.dptr_  + startOffset;
  DType2 *odata = out_data.dptr_ + startOffset;

  for (size_t outer = 0; outer < num; ++outer) {
    for (size_t i = 0; i < matrixSize; ++i) {
      onData(data++, odata++);
    }
    data  += skipLength;
    odata += skipLength;
  }
}

}  // namespace batchnormaddrelu

/*! \brief Forward CPU */
template <typename xpu, typename DType, typename AccReal>
void BatchNormAddReluForwardImpl(mshadow::Stream<cpu> *,
                          const OpContext &ctx, const BatchNormAddReluParam& param_,
                          const std::vector<TBlob> &in_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &out_data,
                          const std::vector<TBlob> &aux_states) {
  // Input
  batchnormaddrelu::BNTensor3<DType> inputData(in_data[batchnormaddrelu::kData], param_.axis);
  const TBlob &weights         = in_data[batchnormaddrelu::kGamma];
  const TBlob &bias            = in_data[batchnormaddrelu::kBeta];

  // Aux (Moving)
  const TBlob &runningMean     = aux_states[batchnormaddrelu::kMovingMean];
  const TBlob &runningVariance = aux_states[batchnormaddrelu::kMovingVar];

  // Output
  batchnormaddrelu::BNTensor3<DType> outputData(out_data[batchnormaddrelu::kOut], param_.axis);
  const TBlob &meanVector      = out_data[batchnormaddrelu::kMean];
  const TBlob &varianceVector  = out_data[batchnormaddrelu::kVar];

  AccReal *mean = meanVector.dptr<AccReal>();
  AccReal  *var = varianceVector.dptr<AccReal>();

  const bool is_train_and_not_global_stats = ctx.is_train && !param_.use_global_stats;
  const size_t channelCount = inputData.ChannelCount();
  const size_t itemCountPerChannel = inputData.Size() / channelCount;

  #pragma omp parallel for
  for (int channel = 0; channel < static_cast<int>(channelCount); ++channel) {
    if (is_train_and_not_global_stats) {
      // compute mean per input
      mean[channel] = 0;
      ForEachFast(inputData, channel, [mean, channel](const DType *in_data) {
        mean[channel] += *in_data; });
      mean[channel] /= itemCountPerChannel;

      // compute variance per input
      const AccReal thisMean = mean[channel];
      var[channel] = 0;
      ForEachFast(inputData, channel,
                  [var, thisMean, channel](const DType *current_in_data) {
                    const AccReal current = *current_in_data;
                    var[channel] += (current - thisMean) * (current - thisMean);
                  });

      const AccReal sum = var[channel];

      AccReal invstd;
      if (sum == 0 && param_.eps == 0.0) {
        // Nobody likes to divide by zero
        invstd = 0;
      } else {
        const AccReal variance = sum / itemCountPerChannel;
        invstd = VARIANCE_TO_INVSTD(variance, param_.eps);
      }
      var[channel] = invstd;
    } else {
      const AccReal *rm = runningMean.dptr<AccReal>();
      const AccReal *rv = runningVariance.dptr<AccReal>();

      mean[channel] = rm[channel];
      var[channel] = VARIANCE_TO_INVSTD(rv[channel], param_.eps);
    }

    // compute output
    AccReal *w = weights.dptr<AccReal>();
    const AccReal *b = bias.dptr<AccReal>();

    const AccReal thisMean = mean[channel];
    const AccReal thisInvstd = var[channel];
    const AccReal thisWeight = w[channel];
    const AccReal thisBias = b[channel];

    // note that var is still invstd
    if (!param_.fix_gamma) {
      if (IsBNAddReluWriting(req[batchnormaddrelu::kData])) {
        ForEachFast(inputData, outputData, channel,
                    [thisWeight, thisBias, thisMean, thisInvstd](const DType *in_data,
                                                                 DType *out_data) {
                      *out_data = static_cast<DType>(
                        ((*in_data - thisMean) * thisInvstd) * thisWeight + thisBias);
                    });
      }
    } else {
      if (IsBNAddReluWriting(req[batchnormaddrelu::kGamma])) {
        w[channel] = AccReal(1);
      }
      if (IsBNAddReluWriting(req[batchnormaddrelu::kData])) {
        ForEachFast(inputData, outputData, channel,
                    [thisWeight, thisBias, thisMean, thisInvstd](const DType *in_data,
                                                                 DType *out_data) {
                      *out_data = static_cast<DType>(
                        ((*in_data - thisMean) * thisInvstd) + thisBias);
                    });
      }
    }
  }
}

template <typename xpu, typename DType, typename AccReal>
void BatchNormAddReluBackwardImpl(mshadow::Stream<cpu> *,
                           const OpContext &ctx, const BatchNormAddReluParam& param_,
                           const std::vector<TBlob> &out_grad,
                           const std::vector<TBlob> &in_data,
                           const std::vector<TBlob> &out_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &in_grad,
                           const std::vector<TBlob> &aux_states) {
  // Input Data
  batchnormaddrelu::BNTensor3<DType> inputData(in_data[batchnormaddrelu::kData], param_.axis);
  const TBlob &weights   = in_data[batchnormaddrelu::kGamma];

  // Input Grad
  batchnormaddrelu::BNTensor3<DType> gradIn(in_grad[batchnormaddrelu::kData], param_.axis);
  const TBlob &gradWeight = in_grad[batchnormaddrelu::kGamma];
  const TBlob &gradBias   = in_grad[batchnormaddrelu::kBeta];

  // Aux (Moving)
  const TBlob &runningMean = aux_states[batchnormaddrelu::kMovingMean];
  const TBlob &runningVariance = aux_states[batchnormaddrelu::kMovingVar];

  // Output
  batchnormaddrelu::BNTensor3<DType> gradOut(out_grad[batchnormaddrelu::kOut], param_.axis);
  const TBlob &saveMean = out_data[batchnormaddrelu::kMean];
  const TBlob &saveStd  = out_data[batchnormaddrelu::kVar];

  const size_t channelCount = inputData.ChannelCount();
  const size_t itemCount    = inputData.Size() / channelCount;

  // Avoid multiple dptr() call within the channel loop
  AccReal *runningMeanDataPtr = runningMean.dptr<AccReal>();
  AccReal *runningVarDataPtr  = runningVariance.dptr<AccReal>();
  const AccReal *saveMeanDataPtr = saveMean.dptr<AccReal>();
  const AccReal *saveInvStdDataPtr = saveStd.dptr<AccReal>();
  AccReal *gradWeightData = gradWeight.dptr<AccReal>();
  AccReal *gradBiasData = gradBias.dptr<AccReal>();

  const bool is_train_and_not_global_stats = ctx.is_train && !param_.use_global_stats;

  #pragma omp parallel for
  for (int channel = 0; channel < static_cast<int>(channelCount); ++channel) {
    const AccReal *weight = weights.dptr<AccReal>();
    const AccReal w = !param_.fix_gamma ? weight[channel] : AccReal(1);
    AccReal mean, invstd;
    if (is_train_and_not_global_stats) {
      mean = saveMeanDataPtr[channel];
      invstd = saveInvStdDataPtr[channel];
      const AccReal variance = INVSTD_TO_VARIANCE(invstd, param_.eps);

      // update running averages
      runningMeanDataPtr[channel] = runningMeanDataPtr[channel] * param_.momentum
                                    + mean * (AccReal(1) - param_.momentum);

      runningVarDataPtr[channel] = runningVarDataPtr[channel] * param_.momentum
                                   + variance * (AccReal(1) - param_.momentum);

    } else {
      mean = runningMeanDataPtr[channel];
      invstd = VARIANCE_TO_INVSTD(runningVarDataPtr[channel], param_.eps);
    }

    // sumGradOut over all gradOutput in feature plane
    AccReal sumGradOut = 0;
    ForEachFast(gradOut, static_cast<size_t>(channel),
                [&sumGradOut](const DType *gradOut_data) {
                  sumGradOut += *gradOut_data;
                });

    // dot product of the Q(X) and gradOuput
    AccReal dotp = 0;
    ForEachFast(inputData, gradOut, static_cast<size_t>(channel),
                [&dotp, mean](const DType *thisInputData, const DType *gradOut_data) {
                  dotp += (*thisInputData - mean) * (*gradOut_data);
                });

    if (!gradIn.IsEmpty() && IsBNAddReluWriting(req[batchnormaddrelu::kData])) {  // grad input?
      if (is_train_and_not_global_stats) {
        // when in training mode
        // Q(X) = X - E[x] ; i.e. input centered to zero mean
        // Y = Q(X) / σ    ; i.e. BN output before weight and bias
        // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ * w

        // projection of gradOutput on to output scaled by std
        const AccReal k = dotp * invstd * invstd / itemCount;
        ForEachFast(inputData, gradIn, static_cast<size_t>(channel),
                    [&mean, &k](const DType *inputDataPtr, DType *gradIn_data) {
                      *gradIn_data = (*inputDataPtr - mean) * k;
                    });

        const AccReal iw = invstd * w;
        const AccReal gradMean = sumGradOut / itemCount;
        ForEachFast(gradOut, gradIn, static_cast<size_t>(channel),
                    [iw, gradMean](const DType *gradOut_data, DType *gradIn_data) {
                      *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * iw;
                    });
      } else {
        // when in evaluation mode
        // Q(X) = X - running_mean  ; i.e. input centered to zero mean
        // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
        // dL/dX = w / running_std
        const AccReal iw = invstd * w;
        ForEachFast(gradOut, gradIn, static_cast<size_t>(channel),
                    [iw](const DType *gradOut_data, DType *gradIn_data) {
                      *gradIn_data = *gradOut_data * iw;
                    });
      }
    }

    // May want to make this a param eventually
    const AccReal scale = 1.0f;

    if (IsBNAddReluWriting(req[batchnormaddrelu::kGamma])) {
      if (!param_.fix_gamma) {
        gradWeightData[channel] = scale * dotp * invstd;
      } else {
        gradWeightData[channel] = AccReal(0);
      }
    }

    if (IsBNAddReluWriting(req[batchnormaddrelu::kBeta])) {
      gradBiasData[channel] = scale * sumGradOut;
    }
  }
}

DMLC_REGISTER_PARAMETER(BatchNormAddReluParam);

static bool BatchNormAddReluShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_shape,
                           std::vector<TShape> *out_shape) {
  const BatchNormAddReluParam& param = nnvm::get<BatchNormAddReluParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 6U) << "Input:[data, gamma, beta, addend, MovingMean, MovingVar]";
  CHECK_EQ(out_shape->size(), 4U);
  SHAPE_ASSIGN_CHECK(*in_shape, batchnormaddrelu::kAddend, in_shape->at(batchnormaddrelu::kData));
  SHAPE_ASSIGN_CHECK(*in_shape, batchnormaddrelu::kData, in_shape->at(batchnormaddrelu::kAddend));
  const TShape &dshape = in_shape->at(batchnormaddrelu::kData);

  const size_t channelAxis = static_cast<size_t>(param.axis < 0
      ? static_cast<int>(dshape.ndim()) + param.axis
      : param.axis);
  CHECK_LT(channelAxis, dshape.ndim()) << "Channel axis out of range: " << param.axis;

  const int channelCount = dshape[channelAxis];

  if (dshape.ndim() == 0) {
    return false;
  }

  in_shape->at(batchnormaddrelu::kGamma) = TShape(Shape1(channelCount));
  in_shape->at(batchnormaddrelu::kBeta) = TShape(Shape1(channelCount));
  in_shape->at(batchnormaddrelu::kInMovingMean) = TShape(Shape1(channelCount));  // kMovingMean
  in_shape->at(batchnormaddrelu::kInMovingVar) = TShape(Shape1(channelCount));  // kMovingVar

  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  out_shape->push_back(Shape1(channelCount));  // kMean
  out_shape->push_back(Shape1(channelCount));  // kVar

  // This is kernel-specific, although it's roughly a bit per input.  This should
  // come directly from the operator implementation.
#define ROUND_UP_TO_MULTIPLE(x, multiple) (((x + multiple - 1) / multiple) * multiple)
  const int nhwCount = dshape.Size()/channelCount;
  const int C_ELEMENTS_PER_CTA = 64;
  const int int32_bits = 32;
  const int effective_c = ROUND_UP_TO_MULTIPLE(channelCount, C_ELEMENTS_PER_CTA);
  const int effective_nhw = ROUND_UP_TO_MULTIPLE(nhwCount, int32_bits);
  const int bitmask_bits = effective_c * effective_nhw;
  const int bitmask_words = bitmask_bits / int32_bits;

  out_shape->push_back(Shape1(bitmask_words));  // kBitmask

  return true;
}

static bool BatchNormAddReluType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_type, std::vector<int> *out_type) {
  using namespace mshadow;
  CHECK_GE(in_type->size(), 4U);
  TYPE_ASSIGN_CHECK(*in_type, batchnormaddrelu::kAddend, in_type->at(batchnormaddrelu::kData));
  TYPE_ASSIGN_CHECK(*in_type, batchnormaddrelu::kData, in_type->at(batchnormaddrelu::kAddend));
  const int dtype = (*in_type)[batchnormaddrelu::kData];
  CHECK_NE(dtype, -1) << "Data or addend input must have a specified type";
  // For float16 input type beta, gamma, mean, and average are stored in float32.
  // For other input types, these parameters have the same type as input
  // NOTE: This requirement is from cuDNN (v. 4 and 5)
  int dtype_param;
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_param = mshadow::DataType<AccRealX>::kFlag; });
  std::vector<std::string> args{"data", "gamma", "beta", "addend", "mean", "var"};
  CHECK_LE(in_type->size(), args.size());
  for (size_t i = 0; i < in_type->size(); ++i) {
    // kData and kAddend have their types established already above
    if (i == batchnormaddrelu::kData || i == batchnormaddrelu::kAddend)
      continue;
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  const size_t n_out = 3;
  out_type->clear();
  out_type->push_back(dtype);            // output
  for (size_t i = 1; i < n_out; ++i) {
    out_type->push_back(dtype_param);    // save_mean, save_var
  }
  out_type->push_back(mshadow::kInt32);  // bitmask
  return true;
}

static inline bool BatchNormAddReluStorageType(const nnvm::NodeAttrs &attrs,
                                        const int dev_mask,
                                        DispatchMode *dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  const BatchNormAddReluParam &param = nnvm::get<BatchNormAddReluParam>(attrs.parsed);

  bool dispatched = false;
  for (int& v : *in_attrs)
    if (v == - 1) v = kDefaultStorage;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  if (!common::ContainsOnlyStorage(*in_attrs, kDefaultStorage) && param.fix_gamma) {
    LOG(FATAL) << "fix_gamma=True is not supported for sparse ndarrays. Tracked at #11647";
  }
  return dispatched;
}

std::vector<nnvm::NodeEntry> BatchNormAddReluGrad(const nnvm::NodePtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> out_data(n->num_outputs());
  for (uint32_t i = 0; i < out_data.size(); ++i) {
    out_data[i] = nnvm::NodeEntry{n, i, 0};
  }
  std::vector<nnvm::NodeEntry> heads;
  heads.reserve(9);
  heads.push_back(ograds[0]);
  heads.push_back(out_data[batchnormaddrelu::kMean]);
  heads.push_back(out_data[batchnormaddrelu::kVar]);
  heads.push_back(out_data[batchnormaddrelu::kBitmask]);
  heads.push_back(n->inputs[batchnormaddrelu::kData]);
  heads.push_back(n->inputs[batchnormaddrelu::kGamma]);
  heads.push_back(n->inputs[batchnormaddrelu::kBeta]);
  heads.push_back(n->inputs[batchnormaddrelu::kInMovingMean]);
  heads.push_back(n->inputs[batchnormaddrelu::kInMovingVar]);

  nnvm::NodePtr gnode = nnvm::Node::Create();
  gnode->inputs = std::move(heads);
  gnode->control_deps.emplace_back(n);
  gnode->attrs = n->attrs;
  gnode->attrs.op = nnvm::Op::Get("_backward_BatchNormAddRelu");
  gnode->attrs.name = n->attrs.name + "_backward";
  // The input of batchnormaddrelu
  std::vector<nnvm::NodeEntry> in_grad(6);
  for (uint32_t i = 0; i < batchnormaddrelu::kInMovingMean; ++i) {
    in_grad[i] = nnvm::NodeEntry{gnode, i, 0};
  }

  // attach no gradient node to forbid gradient on aux_state
  nnvm::NodePtr ng = nnvm::Node::Create();
  ng->attrs.op = Op::Get("_NoGradient");
  ng->attrs.name = "NoGradient";
  // the aux state of batchnormaddrelu
  for (uint32_t i = batchnormaddrelu::kInMovingMean; i <= batchnormaddrelu::kInMovingVar; ++i) {
    in_grad[i] = nnvm::NodeEntry{ng, 0, 0};
  }
  return in_grad;
}

NNVM_REGISTER_OP(BatchNormAddRelu)
.describe(R"code(Batch normalization with built-in addition and ReLU Activation.

Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.  This version is somewhat special purpose in that the
usual normalized data output is then added to an additional data input, followed
by ReLU activation.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis:

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these
two outputs are blocked.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

If ``use_global_stats`` is set to be true, then ``moving_mean`` and
``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
the output. It is often used during inference.

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
axis to be the last item in the input shape.

Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
then set ``gamma`` to 1 and its gradient to 0.

Note::

When fix_gamma is set to True, no sparse support is provided. If fix_gamma is set to False,
the sparse tensors will fallback.

)code" ADD_FILELINE)
.set_num_inputs(6)
.set_num_outputs(4)
.set_attr_parser(ParamParser<BatchNormAddReluParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta", "addend", "moving_mean", "moving_var"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "var", "bitmask"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const BatchNormAddReluParam& param = nnvm::get<BatchNormAddReluParam>(attrs.parsed);
  // Bitmask is always invisible
  return param.output_mean_var ? 3 : 1;
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const nnvm::NodeAttrs& attrs) {
  return std::vector<uint32_t>{batchnormaddrelu::kInMovingMean, batchnormaddrelu::kInMovingVar};
})
.set_attr<mxnet::FInferShape>("FInferShape", BatchNormAddReluShape)
.set_attr<nnvm::FInferType>("FInferType", BatchNormAddReluType)
.set_attr<FCompute>("FCompute<cpu>", BatchNormAddReluCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", BatchNormAddReluGrad)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  // input 3 (addend) can share its tensor with output 0 (y)
  return std::vector<std::pair<int, int> >{{3, 0}};
})
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("addend", "NDArray-or-Symbol", "input summed with BN output before relu")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(BatchNormAddReluParam::__FIELDS__())
.set_attr<nnvm::FSetInputVarAttrOnCompose>(
  "FSetInputVarAttrOnCompose",
  [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
    if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
    if (index == batchnormaddrelu::kInMovingMean) {
      var->attrs.dict["__init__"] = "[\"zero\", {}]";
    } else if (index == batchnormaddrelu::kInMovingVar) {
      var->attrs.dict["__init__"] = "[\"one\", {}]";
    }
  });

NNVM_REGISTER_OP(_backward_BatchNormAddRelu)
.set_num_inputs(9)
.set_num_outputs(4)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  // input 4 (x input) can share its tensor with output 0 (dx).
  // input 0 (y gradient) can share its tensor with output 0 (dx)
  return std::vector<std::pair<int, int> >{{0, 0}, {4, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(ParamParser<BatchNormAddReluParam>)
.set_attr<FCompute>("FCompute<cpu>", BatchNormAddReluGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
