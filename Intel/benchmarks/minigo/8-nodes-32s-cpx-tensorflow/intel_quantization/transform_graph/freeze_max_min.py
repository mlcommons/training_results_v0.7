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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes


def parse_input_graph(input_graph_def):
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            print('Duplicate node name {}'.format(node.name))
    return input_node_map


def get_valid_log(max_min_log):
    with open(max_min_log) as f:
        lines = f.readlines()
    output = []
    target_lines = [i.strip() for i in lines if i.strip().find(';') != -1]
    for i in target_lines:
        semi_count = i.count(';')
        if semi_count == 2:
            output.append(i)
        elif semi_count % 2 != 0:
            print("Invalid line")
        else:
            loop_times = int(semi_count / 2)
            semi_index = [index for index, value in enumerate(i) if value == ";"]
            for index in range(loop_times - 1):
                output.append(i[semi_index[index * 2]: semi_index[index * 2 + 2]])
            output.append(i[semi_index[loop_times * 2 - 2]:])
    return output


def parse_requantization_ranges(max_min_log):
    """
    Parse the max_min log to get requantization values
    :param max_min_log: input min max log file
    :return: dict saved the result
    """
    print_suffix = "__print__"
    post_fix = "__requant_min_max"
    lines = get_valid_log(max_min_log)
    res = {}
    temp_min = {}
    temp_max = {}
    for i in lines:
        if i.find(print_suffix + ";" + post_fix) == -1:
            continue
        max_line_data = i.split(print_suffix + ";" + post_fix)[-1]
        min_value = max_line_data.split('][')[0].split('[')[1]
        max_value = max_line_data.split('][')[1].split(']')[0]
        name = i.split(';')[1].strip()[:-len(print_suffix)]
        if name not in temp_min:
            temp_min[name] = []
        if name not in temp_max:
            temp_max[name] = []

        temp_min[name].append(float(min_value))
        temp_max[name].append(float(max_value))

    for key in temp_min:
        target_min_index = int(round(len(temp_min[key]) * 0.05))
        if target_min_index < 0:
            target_min_index = 0
        if key not in res:
            res[key] = []
        res[key].append(sorted(temp_min[key])[target_min_index])
    for key in temp_max:
        target_max_index = int(round(len(temp_max[key]) * 0.95))
        if target_max_index > len(temp_max[key]) - 1:
            target_max_index = len(temp_max[key]) - 1
        res[key].append(sorted(temp_max[key])[target_max_index])
    return res


def parse_max_min_log(max_min_log, fetch_max=True):
    """
    Parse the max_ming log file
    :param max_min_log: max_min log file
    :param fetch_max: parse for freeze_max or not
    :return: get the node name and value mapping
    """
    print_suffix = "__print__"
    if fetch_max:
        postfix = "__max:"
    else:
        postfix = "__min:"

    lines = get_valid_log(max_min_log)

    res = {}
    temp = {}
    for i in lines:
        if i.find(print_suffix + ";" + postfix) == -1:
            continue
        max_line_data = i.split(';')
        name = max_line_data[1][:-len(print_suffix)]
        value = max_line_data[-1].split('[')[-1].split(']')[0]
        if "eightbit" in name and name not in temp:
            temp[name] = []
        if "eightbit" in name:
            temp[name].append(float(value))
    for key in temp:
        target_index = int(len(temp[key]) * 0.95)
        if target_index > len(temp[key]) - 1:
            target_index = len(temp[key]) - 1
        res[key] = sorted(temp[key])[target_index]
    return res


def generate_output_graph_ranges(input_node_map, range_info):
    output_graph_def = graph_pb2.GraphDef()
    inputs_to_rename = {}
    for node in input_node_map:
        if node in range_info:
            min_node = node_def_pb2.NodeDef()
            min_node.op = "Const"
            min_node.name = node + "/frozen_min"
            inputs_to_rename[node + ":0"] = min_node.name + ":0"
            min_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            min_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(float(range_info[node][0]), dtypes.float32, [])))

            max_node = node_def_pb2.NodeDef()
            max_node.op = "Const"
            max_node.name = node + "/frozen_max"
            inputs_to_rename[node + ":1"] = max_node.name + ":0"
            max_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            max_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(float(range_info[node][1]), dtypes.float32, [])))
            output_graph_def.node.extend([min_node, max_node])
        else:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(input_node_map[node])
            output_graph_def.node.extend([new_node])

    for node in output_graph_def.node:
        found_index = []

        for input_index, input_name in enumerate(node.input):
            for _, v in enumerate(inputs_to_rename):
                if input_name == v:
                    found_index.append(input_index)

        if found_index:
            for sub_index in found_index:
                node.input[sub_index] = inputs_to_rename[node.input[sub_index]]

    return output_graph_def


def generate_output_graph(input_node_map, max_name_value, is_max=True):
    """
    Generate transformed graph for freeze_max/freeze_min transformation.
    :param input_node_map: input node name and nodedef mapping
    :param max_name_value: target values
    :param is_max: freeze_max flag
    :return: transformed graph
    """
    output_graph_def = graph_pb2.GraphDef()
    inputs_to_rename = {}
    for node in input_node_map:
        if node in max_name_value:
            new_node = node_def_pb2.NodeDef()
            new_node.op = "Const"
            new_node_postfix = "/frozen_max_only" if is_max else "/frozen_min_only"
            new_node.name = node + new_node_postfix
            inputs_to_rename[node] = new_node.name + ":0"
            new_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            new_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(float(max_name_value[node]), dtypes.float32, [])))
        else:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(input_node_map[node])
        output_graph_def.node.extend([new_node])

    for node in output_graph_def.node:
        found = False
        found_index = -1
        found_value = ""
        for input_index, input_name in enumerate(node.input):
            for _, v in enumerate(inputs_to_rename):
                if input_name == v:
                    found = True
                    found_index = input_index
                    found_value = v
                    break
            if found:
                break
        if found:
            post_fix = '/frozen_max_only:0' if is_max else '/frozen_min_only:0'
            node.input[found_index] = found_value + post_fix

    return output_graph_def


def freeze_requantization_range(input_graph_def, max_min_log):
    """
    Freeze requantization range graph transformation
    :param input_graph_def: input graphdef
    :param max_min_log: max_min_log file
    :return: transformed graph
    """
    input_node_map = parse_input_graph(input_graph_def)
    range_info = parse_requantization_ranges(max_min_log)
    return generate_output_graph_ranges(input_node_map, range_info)


def freeze_max(input_graph_def, max_min_log):
    """
    Freeze max graph transformation
    :param input_graph_def: input graphdef
    :param max_min_log: max_min_log
    :return: transformed graph
    """
    input_node_map = parse_input_graph(input_graph_def)
    max_name_value = parse_max_min_log(max_min_log, True)
    return generate_output_graph(input_node_map, max_name_value, True)


def freeze_min(input_graph_def, max_min_log):
    """
    Freeze min graph transformation.
    :param input_graph_def: input graphdef
    :param max_min_log: max_min_log file
    :return: transformed graph
    """
    input_node_map = parse_input_graph(input_graph_def)
    max_name_value = parse_max_min_log(max_min_log, False)
    return generate_output_graph(input_node_map, max_name_value, False)
