#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
# from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

# from intel_quantization.quantize_graph import GraphRewriter
from intel_quantization.transform_graph.strip_unused import StripUnusedNodes
from intel_quantization.transform_graph.fold_batch_norm import FoldBatchNormNodes
from intel_quantization.transform_graph.insert_logging import InsertLogging
from intel_quantization.transform_graph.freeze_max_min import freeze_max
from intel_quantization.transform_graph.freeze_max_min import freeze_min
from intel_quantization.transform_graph.freeze_max_min import freeze_requantization_range
from intel_quantization.transform_graph.fuse_quantized_conv_and_requantize import fuse_quantized_conv_and_requantize
from intel_quantization.transform_graph.fuse_quantized_mul_and_requantize import FuseQuantizedMulAndRequantize
from intel_quantization.transform_graph.fuse_column_wise_mul import FuseColumnWiseMul
from intel_quantization.transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from intel_quantization.util import read_graph, write_graph
from intel_quantization.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
import os
import shlex
import subprocess
import sys
import logging

logging.getLogger().setLevel(level=logging.INFO)

tf.compat.v1.disable_eager_execution()

TF_SUPPORTED_MAX_VERSION = '2.2.0'
TF_SUPPORTED_MIN_VERSION = '1.14.0'


class GraphConverter:
    def __init__(self, input_graph, output_graph, inputs=[], outputs=[], excluded_ops=[], excluded_nodes=[],
                 per_channel=False, input_graph_is_binary=True):
        """Convert graph.

        :param input_graph: input graph pb file.
        :param output_graph: output graph pb file. If set, output directory should be exist.
        :param inputs: input nodes' names.
        :param outputs: output nodes' names.
        :param excluded_ops: list of operations to be excluded from quantization.
        :param excluded_nodes: list of nodes to be excluded from quantization.
        :param per_channel: if set True, enables weight quantization channel-wise.
        :param input_graph_is_binary: default True, whether input graph is binary.
        """
        self.input_graph = input_graph
        self.input_graph_binary_flag = input_graph_is_binary
        self.output_graph = output_graph
        self.inputs = inputs
        self.outputs = outputs
        # quantize specific config
        self.per_channel = per_channel
        self.excluded_ops = excluded_ops
        self.excluded_nodes = excluded_nodes
        self._low_precision_mode = 'eightbit'

        self.gen_calib_data_cmds = None
        self.debug = False
        self._check_tf_version()
        self._check_args()
        self._gen_tmp_filenames()

    def _check_tf_version(self):
        is_supported_version = False
        try:
            from tensorflow import python
            if (hasattr(python, "pywrap_tensorflow") and hasattr(python.pywrap_tensorflow, "IsMklEnabled")):
                from tensorflow.python.pywrap_tensorflow import IsMklEnabled
            else:
                from tensorflow.python._pywrap_util_port import IsMklEnabled
            if IsMklEnabled() and (TF_SUPPORTED_MIN_VERSION <= tf.__version__ <= TF_SUPPORTED_MAX_VERSION):
                is_supported_version = True
        except Exception as e:
            raise ValueError(e)
        finally:
            if not is_supported_version:
                raise ValueError(str('Please install Intel® Optimizations for TensorFlow'
                                     ' or MKL enabled source build TensorFlow'
                                     ' with version >={} and <={}').format(TF_SUPPORTED_MIN_VERSION,
                                                                           TF_SUPPORTED_MAX_VERSION))

    def _check_args(self):
        if not gfile.Exists(self.input_graph):
            raise ValueError('Input graph pb file %s does not exist.' % self.input_graph)
        if self.output_graph and not os.path.exists(os.path.dirname(self.output_graph)):
            raise ValueError('"output_graph" directory does not exist.')

        self._output_path = os.path.dirname(os.path.realpath(self.output_graph if self.output_graph
                                                             else self.input_graph))

    def _gen_tmp_filenames(self):
        self._fp32_optimized_graph = os.path.join(self._output_path, 'fp32_optimized_graph.pb')
        self._int8_dynamic_range_graph = os.path.join(self._output_path, 'int8_dynamic_range_graph.pb')
        self._int8_logged_graph = os.path.join(self._output_path, 'int8_logged_graph.pb')
        self._requant_min_max_log = os.path.join(self._output_path, 'requant_min_max_log.txt')
        self._int8_frozen_range_graph = os.path.join(self._output_path, 'int8_frozen_range_graph.pb')
        if not self.output_graph:
            self.output_graph = os.path.join(self._output_path, 'int8_final_fused_graph.pb')
        # to keep temp graphDef
        self._tmp_graph_def = None

    def convert(self):
        """Do convert, including:
            1) optimize fp32_frozen_graph,
            2) quantize graph,
            3) calibration,
            4) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        try:
            self._optimize_frozen_fp32_graph()
        except Exception as e:
            logging.error('Failed to optimize fp32 graph due to: %s', str(e))
            raise ValueError(e) from e
        else:
            self.quantize()

    def quantize(self):
        """Quantize graph only (without optimizing fp32 graph), including:
            1) quantize graph,
            2) calibration,
            3) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        if not self.gen_calib_data_cmds:
            raise ValueError('Pass an inference command for accuracy to "gen_calib_data_cmds" '
                             'to generate calibration data.')
        try:
            self._quantize_graph()
            self._insert_logging()
            self._generate_calibration_data()
            self._freeze_requantization_ranges()
            self._fuse_quantized_mul_and_requantize()
            self._fuse_requantize_with_fused_quantized_conv()
        except Exception as e:
            logging.error('Failed to quantize graph due to: %s', str(e))
            raise ValueError(e) from e
        finally:
            if not self.debug:
                self._post_clean()

    def _optimize_frozen_fp32_graph(self):
        """Optimize fp32 frozen graph."""

        self._tmp_graph_def = read_graph(self.input_graph, self.input_graph_binary_flag)
        dtypes = self._get_dtypes(self._tmp_graph_def)
        # self._tmp_graph_def = optimize_for_inference(self._tmp_graph_def, self.inputs, self.outputs, dtypes, False)
        self._tmp_graph_def = FuseColumnWiseMul(self._tmp_graph_def).do_transformation()
        self._tmp_graph_def = StripUnusedNodes(self._tmp_graph_def, self.inputs, self.outputs, dtypes).do_transform()
        self._tmp_graph_def = graph_util.remove_training_nodes(self._tmp_graph_def, self.outputs)
        self._tmp_graph_def = FoldBatchNormNodes(self._tmp_graph_def).do_transform()
        write_graph(self._tmp_graph_def, self._fp32_optimized_graph)

    def _quantize_graph(self):
        """quantize graph."""

        if not self._tmp_graph_def:
            self._tmp_graph_def = read_graph(self.input_graph, self.input_graph_binary_flag)

        g = ops.Graph()
        with g.as_default():
            importer.import_graph_def(self._tmp_graph_def)

        intel_quantizer = QuantizeGraphForIntel(self._tmp_graph_def,
                                                self.outputs, self.per_channel,
                                                excluded_ops=self.excluded_ops,
                                                excluded_nodes=self.excluded_nodes)
        self._tmp_graph_def = intel_quantizer.do_transform()

        if self.debug:
            write_graph(self._tmp_graph_def, self._int8_dynamic_range_graph)

    def _insert_logging(self):
        int8_dynamic_range_graph_def = graph_pb2.GraphDef()
        int8_dynamic_range_graph_def.CopyFrom(self._tmp_graph_def)
        InsertLogging(self._tmp_graph_def,
                      ops=["RequantizationRange{}".format("PerChannel" if self.per_channel else "")],
                      message="__requant_min_max:").do_transformation()
        InsertLogging(self._tmp_graph_def, ops=["Min"], message="__min:").do_transformation()
        InsertLogging(self._tmp_graph_def, ops=["Max"], message="__max:").do_transformation()
        write_graph(self._tmp_graph_def, self._int8_logged_graph)
        self._tmp_graph_def.CopyFrom(int8_dynamic_range_graph_def)

    def _generate_calibration_data(self):
        cmd = self.gen_calib_data_cmds
        cmd = cmd.format(self._int8_logged_graph)
        f = open(self._requant_min_max_log, 'w', buffering=1)
        p = subprocess.Popen(shlex.split(cmd), stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        try:
            for line in p.stdout:
                line_str = line.decode(sys.stdout.encoding)
                sys.stdout.write(line_str)
                f.write(line_str)
            p.communicate()
        except Exception:
            p.kill()
            p.wait()
            raise
        if p.poll():
            raise SystemExit('ERROR generating calibration data, command: \n{}'.format(cmd))

    def _freeze_requantization_ranges(self):
        self._tmp_graph_def = freeze_max(self._tmp_graph_def, self._requant_min_max_log)
        self._tmp_graph_def = freeze_min(self._tmp_graph_def, self._requant_min_max_log)
        self._tmp_graph_def = freeze_requantization_range(self._tmp_graph_def, self._requant_min_max_log)
        if self.debug:
            write_graph(self._tmp_graph_def, self._int8_frozen_range_graph)

    def _fuse_quantized_mul_and_requantize(self):
        self._tmp_graph_def = FuseQuantizedMulAndRequantize(self._tmp_graph_def).do_transformation()


    def _fuse_requantize_with_fused_quantized_conv(self):
        self._tmp_graph_def = fuse_quantized_conv_and_requantize(self._tmp_graph_def)
        # strip_unused_nodes with optimize_for_inference
        dtypes = self._get_dtypes(self._tmp_graph_def)
        # self._tmp_graph_def = optimize_for_inference(self._tmp_graph_def, self.inputs, self.outputs, dtypes, False)
        self._tmp_graph_def = StripUnusedNodes(self._tmp_graph_def, self.inputs, self.outputs, dtypes).do_transform()
        self._tmp_graph_def = graph_util.remove_training_nodes(self._tmp_graph_def, self.outputs)
        self._tmp_graph_def = FoldBatchNormNodes(self._tmp_graph_def).do_transform()
        RerangeQuantizedConcat(self._tmp_graph_def).do_transformation()
        write_graph(self._tmp_graph_def, self.output_graph)
        logging.info('Converted graph file is saved to: %s', self.output_graph)

    def _get_dtypes(self, in_graph_def):
        # TODO: keep dtypes list order as input list?
        dtypes = []
        for n in in_graph_def.node:
            if n.name in self.inputs:
                dtypes.append(n.attr["dtype"].type)

        return dtypes

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None
        """
        if gfile.Exists(self._int8_logged_graph):
            os.remove(self._int8_logged_graph)
        if gfile.Exists(self._requant_min_max_log):
            os.remove(self._requant_min_max_log)
