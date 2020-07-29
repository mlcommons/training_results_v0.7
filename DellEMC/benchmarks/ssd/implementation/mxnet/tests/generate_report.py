#!/usr/bin/env python
import os
import re
import argparse
from glob import glob
import collections

import git
import numpy as np
import pandas as pd
from jinja2 import Template

import plotly
import plotly.express as px
import plotly.graph_objects as go


try:
    git_sha = git.Repo(search_parent_directories=True).head.object.hexsha
    git_sha_short = git_sha[0:5]
except:
    git_sha = '354127fbd3c12410249524617fa18d5e66c58d09'
    git_sha_short = git_sha[0:5]
template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_template.html.jinja2')

parser = argparse.ArgumentParser(description='Generate report for mxnet SSD tests')
parser.add_argument('--input-dir', type=str, default='tests/results/{git_hash}'.format(git_hash=git_sha_short),
                    help='Input directory')
parser.add_argument('--output', type=str, default='report_{git_hash}.html'.format(git_hash=git_sha_short),
                    help='Output HTML file name')
args = parser.parse_args()

def parse_log(fname, mlperf_re=True, target_map=23.0):
    log = open(fname).read()
    train_re = r"\[Training\]\[Epoch (.*)\] training time: (.*) \[sec\], avg speed: (.*) \[imgs/sec\], loss=(.*)"
    val_re = r".*eval_accuracy\", \"value\": (\d*.\d*).*epoch_num\": (\d*)"

    train_matches = re.finditer(train_re, log, re.MULTILINE)
    val_matches = re.finditer(val_re, log, re.MULTILINE)
    
    train_epoch = []
    train_time = []
    train_throughput = []
    train_loss = []
    for match in train_matches:
        train_epoch.append(int(match.group(1)))
        train_time.append(float(match.group(2)))
        train_throughput.append(float(match.group(3)))
        train_loss.append(float(match.group(4)))
    train_df = pd.DataFrame({'epoch': train_epoch,
                             'time': train_time,
                             'throughput': train_throughput,
                             'loss': train_loss,
                             'fname': os.path.basename(fname)})
    val_epoch = []
    val_map = []
    for match in val_matches:
        if mlperf_re:
            val_epoch.append(int(match.group(2)))
            val_map.append(100*float(match.group(1)))
        else:
            val_epoch.append(int(match.group(1)))
            val_map.append(float(match.group(2)))
    val_df = pd.DataFrame({'epoch': val_epoch, 'map': val_map, 'fname': os.path.basename(fname)})

    result = False
    if not val_map or any(np.isnan(x) for x in train_loss):
        result = 'nan'
    elif val_map[-1]>=target_map:
        result = 'success'
    elif val_map[-1]<target_map:
        result = 'failure'
    return train_df, val_df, result


def parse_logs(logs, mlperf_re=True):
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    result = np.array([], dtype=str)
    for log in logs:
        train_df_, val_df_, result_ = parse_log(log, mlperf_re)
        train_df = train_df.append(train_df_, ignore_index=True, sort=False)
        val_df = val_df.append(val_df_, ignore_index=True, sort=False)
        result = np.append(result, result_)
    return train_df, val_df, result


def summarize_test(name, train_df, val_df, result):
    result_collection = collections.Counter(result)
    summary = {}
    summary['name'] = name
    summary['count'] = len(result)
    summary['success'] = result_collection['success']
    summary['nans'] = result_collection['nan']
    summary['failed'] = result_collection['failure']
    summary['non_convergent'] = result_collection['failure']+result_collection['nan']
    
    # Job results bar graph
    results_bar_fig = go.Figure(data=[go.Bar(x=list(result_collection.keys()),
                                              y=list(result_collection.values()),
                                              text=list(result_collection.values()),
                                              textposition='auto')])

    # Successful jobs convergance epoch bar graph
    last_map = val_df.groupby(['fname']).last()
    success_df = last_map[last_map['map']>=23]  # Filter successful jobs only
    agg_successl_df = success_df.groupby('epoch', as_index=False).count()
    x = agg_successl_df['epoch'].values
    y = agg_successl_df['map'].values
    epochs_fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, width=0.4, textposition='auto')])
    epochs_fig.update_layout(barmode='group')

    # Loss and mAP line graphs
    train_loss_fig = None if train_df.empty else px.line(train_df, x="epoch", y="loss", color='fname')
    train_throughput_fig = None if train_df.empty else px.line(train_df, x="epoch", y="throughput", color='fname')
    val_map_fig =  None if val_df.empty else px.line(val_df, x="epoch", y="map", color='fname')

    summary['plot_names'] = ['Results', 'Convergance Distribution', 'Training Loss Graph', 'Validatino mAP Graph', 'Training throughput']
    summary['plots'] = [results_bar_fig, epochs_fig, train_loss_fig, val_map_fig, train_throughput_fig]
    summary['divs'] = []
    for plot in summary['plots']:
        if plot is not None:
            summary['divs'].append(plotly.offline.plot(plot, output_type='div', include_plotlyjs=False))
        else:
            summary['divs'].append("<div>No validation graph</div>")
    return summary


results = []
for config in sorted(glob(os.path.join(args.input_dir, '*'))):
    print("Parsing {}".format(config))
    name = os.path.basename(config)
    logs = glob(os.path.join(config, '*.out'))
    train_df, val_df, success = parse_logs(logs=logs, mlperf_re=True)
    result = summarize_test(name, train_df, val_df, success)
    results.append(result)

title = 'SSD MxNet Test Report'
subtitle = '{input_folder}'.format(input_folder=os.path.basename(args.input_dir))
try:
    Template(open(template_file).read()) \
        .stream(title=title, subtitle=subtitle, tests=results) \
        .dump(args.output)
    print("Wrote report to: {}".format(args.output))
except:
    print("Somthing went wrong, no report was generated")
