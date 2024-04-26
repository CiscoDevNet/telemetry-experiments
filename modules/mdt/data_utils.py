"""
Copyright (c) 2024 Cisco and/or its affiliates.
This software is licensed to you under the terms of the Cisco Sample
Code License, Version 1.1 (the "License"). You may obtain a copy of the
License at
               https://developer.cisco.com/docs/licenses
All use of the material herein must be in accordance with the terms of
the License. All rights not expressly granted by the License are
reserved. Unless required by applicable law or agreed to separately in
writing, software distributed under the License is distributed on an "AS
IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
"""

import numpy as np
from shutil import copyfile
import csv
import re
from collections import defaultdict, deque
from typing import Tuple
from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LinearRegression

def minmax(d, axis=0):
    """"min-max scaler"""
    ran = np.asarray(np.max(d, axis=axis) - np.min(d, axis=axis)).reshape(-1)
    zero_idx = np.argwhere(ran == 0)
    ran[zero_idx] = 1
    return (d-np.min(d, axis=axis)) / ran


def diff(d):
    """perform order 1 diff only when the the values are strictly in/de-cremental along the specified axis

    Note:
        along the column
    """
    d_diff = np.diff(d, axis=0)
    mono_idx = np.concatenate((np.where(np.all(d_diff >= 0, axis=0))[0],
                              np.where(np.all(d_diff <= 0, axis=0))[0]))
    d = d[1:, ]   # remove the first row to be inline with the diffed data
    d[:, mono_idx] = d_diff[:, mono_idx]
    return d


def axis_shift(d):
    """shift each dim by a different lag to avoid having clusters sitting on y=x, y=-x """
    shift = np.min(d, axis=0)
    for dim in range(d.shape[1]):
        d[:, dim] = d[:, dim] - shift[dim] + 1
    return d


def axis_rotate(d):
    """rotate 2-d data by a specific angle"""
    if d.shape[1] != 2:
        return d
    solver = LinearRegression().fit(d[:,0][:, np.newaxis], d[:,1])
    theta = np.pi/4 + np.arctan(solver.coef_)
    # clockwise rotation by theta |\
    rotation_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).reshape((2,2))
    # d = (rotation_mat @ d.T).T
    d = np.dot(rotation_mat, d.T).T
    solver = LinearRegression().fit(d[:,0][:, np.newaxis], d[:,1])
    return d


def scale_data(d):
    d = d - np.mean(d, axis=0)
    ft_scale = np.std(d, axis=0)
    z_index = np.where(ft_scale < 1e-6)
    ft_scale[z_index] = 1
    d = d / ft_scale
    return d

def scale_avg(d):
    """
    Does the same scaling as the option "Scale=Avg" in vis_server GUI
    """
    d = np.abs(d)
    ft_scale = np.mean(d, axis=0)
    z_index = np.where(ft_scale < 1e-6)
    ft_scale[z_index] = 1
    d = d / ft_scale
    return d


def save_IMFs_into_CSV(path,imfs,headers):
    with open(path, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(headers) # write the header
        # write the actual content
        for l in imfs:
            writer.writerow(l)


def get_feature_names(path):
    with open(path, "r") as f:
        d_reader = csv.DictReader(f)
        #get fieldnames from DictReader object and store in list
        headers = d_reader.fieldnames
    return headers


def get_feature_names_bis(path, delimiter=','):
    "a more direct and simpler implementation than get_feature_names()"
    with open(path, "r") as f:
        header = f.readline().strip('\n')
    return header.split(delimiter)


def ft_group_by_yang(feature_names):
    """group features by yang model/sensor path

    a feature name in merged.csv/offline_preprocess.csv looks like the following:
    n0:Cisco-IOS-XR-drivers-media-eth-oper:ethernet-interface_statistics_statistic.csv:HundredGigE0/0/0/0:received-good-bytes

    where "n0" is local identifier for a sensor-path configured in collection,
    "Cisco-IOS-XR-drivers-media-eth-oper:ethernet-interface_statistics_statistic" is
    the sensor path corresponds to "n0".
    in order to group by sensor path, we use a dictionary where the key is the sensor path or its local ID,
    element is the feature indexes under same key

    Args:
        fn (string): files with feature name row

    Returns:
        {"sensor path name": [int, ...],}
    """
    feature_group = defaultdict(list)
    for i, v in enumerate(feature_names):
        k = v.split('.')[0]
        feature_group[k].append(i)
    return feature_group


def ft_dissect(ft_string):
    """given feature name string or column name, try dissect it into module, encoding path, key value array, leafname"""
    subs = deque(ft_string.strip().split(':'))
    leaf = subs.pop()
    if '[' in ft_string:
        # the onbox dataset case
        sp = ft_string.split('[')[0]
        kv = ft_string[ft_string.find("[")+1:ft_string.find("]")].split(':')
        leaf = ft_string.split(']')[-1].split('/')[-1]
        ymd = sp.split(":")[0]
        sp = sp.split(":")[-1]
    else:
        while True:
            try:
                ymd = subs.popleft()
            except IndexError:  # queue empty
                ymd = ''
                break
            if "Cisco" in ymd:
                break
        sp = subs.popleft() if len(subs) > 0 else ''
        kv = list(subs)
    return ymd, sp, kv, leaf.split('/')[-1]



def realdata_loader(fn, scale=False, ft_regex=None, ft_out=False):
    """loader for merged.csv"""
    d = np.genfromtxt(fn, dtype=float, delimiter=',', skip_header=1)
    print("Raw data shape (timestamp col incl.): " + str(d.shape))
    tstp = d[:, 0]

    d = d[:, 1:]
    ft_names = np.asarray(get_feature_names_bis(fn)[1:])
    if ft_regex:
        ft_filter = re.compile(ft_regex, re.IGNORECASE)
        ft_idx = np.array([i for i, v in enumerate(map(ft_filter.match, ft_names)) if v is not None])
        if len(ft_idx) > 0:
            d = d[:, ft_idx]
            ft_names = ft_names[ft_idx]
        else:
            d = np.array([])
            ft_names = np.array([])
        print("Data shape after ft filter (timestamp excl.): " + str(d.shape))

    if scale:
        d = scale_data(d)

    if ft_out:
        return tstp.tolist(), d, ft_names
    else:
        return tstp.tolist(), d


MIN_TIMESTAMP = -62135596800
MAX_TIMESTAMP = 253402214400

ORIGINAL_DATA     = "original data"
REDUCED_DATA      = "reduced data"
FIRST_DERIVATIVE  = "first derivative"
SECOND_DERIVATIVE = "second derivative"

def load_data(in_fn, reduced=None, startTime=MIN_TIMESTAMP, endTime=MAX_TIMESTAMP, 
              scale=False, data_selection={}, ft_regex=None, remove_nan=False, remove_inf=False) -> Tuple[np.array, pd.DataFrame]:
        data = np.genfromtxt(in_fn, dtype=float, delimiter=',', skip_header=1)

        if isinstance(data_selection, str):
            selection = {
                ORIGINAL_DATA    : False,
                REDUCED_DATA     : False,
                FIRST_DERIVATIVE : False,
                SECOND_DERIVATIVE: False
            }
            selection[data_selection] = True
            data_selection = selection

        tstp = data[:,0]
        data = data[:,1:]
        ft_names = np.asarray(get_feature_names_bis(in_fn)[1:])
        if ft_regex:
            ft_filter = re.compile(ft_regex, re.IGNORECASE)
            ft_idx = np.array([i for i, v in enumerate(map(ft_filter.match, ft_names)) if v is not None])
            if len(ft_idx) > 0:
                data = data[:, ft_idx]
                ft_names = ft_names[ft_idx]
            else:
                data = np.array([])
                ft_names = np.array([])

        if remove_nan:
            inval_col = np.where(np.any(np.isnan(data), axis=0))
            data = np.delete(data, inval_col, axis=1)
            ft_names = np.delete(ft_names, inval_col)

        if remove_inf:
            inval_col = np.where(np.any(np.isinf(data), axis=0))
            data = np.delete(data, inval_col, axis=1)
            ft_names = np.delete(ft_names, inval_col)
        
        if scale:
            data = scale_data(data)
        
        final_names = np.asarray([])
        final_data = np.array([[] for _ in range(len(data))])
        derivative = None
        if data_selection[FIRST_DERIVATIVE] or data_selection[SECOND_DERIVATIVE]:
            derivative = np.diff(data, axis=0)

        if data_selection[ORIGINAL_DATA]:
            final_data = np.append(final_data, data, axis=1)
            final_names = np.append(final_names, ft_names)
        
        if data_selection[REDUCED_DATA]:
            final_data = np.append(final_data, reduced, axis=1)
            final_names = np.append(final_names, [f"{x}_bytes-sent_reduced" for x in range(len(reduced[0]))])

        if data_selection[FIRST_DERIVATIVE]:
            final_data = np.append(final_data, np.vstack([derivative[0,:], derivative]), axis=1)
            final_names = np.append(final_names, [f"{x}_bytes-send_deriv" for x in ft_names])

        if data_selection[SECOND_DERIVATIVE]:
            second_derivative = np.diff(derivative, axis=0)
            second_derivative = np.vstack([second_derivative[0,:], second_derivative[0,:], second_derivative])
            final_data = np.append(final_data, second_derivative, axis=1)
            final_names = np.append(final_names, [f"{x}_bytes-sent_deriv2" for x in ft_names])

        # add timestamp            
        final_data = np.append(tstp.reshape(-1,1), final_data, axis=1)
        final_names = np.append(np.asarray('ts'), final_names)

        # filter by time
        if isinstance(startTime, datetime):
            startTime = startTime.replace(tzinfo=timezone.utc).timestamp()
        if isinstance(endTime, datetime):
            endTime = endTime.replace(tzinfo=timezone.utc).timestamp()
        final_data = final_data[
            (final_data[:,0] >= startTime) &
            (final_data[:,0] <= endTime)
        ]
        final_tstp = final_data[:,0]

        return final_tstp, pd.DataFrame(final_data, columns=final_names)

def CRFT_reader(fn, ft_regex=None):
    data, ft_names = [],[]
    with open(fn, newline='') as f:
        c = 0
        for row in csv.reader(f):
            if c == 0:
                tstp = list(map(float, row[1:]))
            else:
                data.append(list(map(int, row[1:])) )
                ft_names.append(row[0])
            c = c+1
    ft_names = np.asarray(ft_names)
    data = np.asarray(data).T
    if ft_regex:
        ft_filter = re.compile(ft_regex, re.IGNORECASE)
        ft_idx = np.array([i for i, v in enumerate(map(ft_filter.match, ft_names)) if v is not None])
        if len(ft_idx) > 0:
            data= data[:, ft_idx]
            ft_names = ft_names[ft_idx]
        print("Data shape after ft filter (timestamp excl.): " + str(data.shape))

    print('CRFT loaded data shape: ', data.shape)
    return ft_names, tstp, data


def clean_nan_inf(data: np.ndarray, ft_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inval_col = np.where(np.any(np.isnan(data), axis=0))
    data = np.delete(data, inval_col, axis=1)
    ft_names = np.delete(ft_names, inval_col)
    print(f'Data shape after nan removal: {data.shape}')

    inval_col = np.where(np.any(np.isinf(data), axis=0))
    data = np.delete(data, inval_col, axis=1)
    ft_names = np.delete(ft_names, inval_col)
    print(f'Data shape after inf removal: {data.shape}')

    return data, ft_names

def adaptive_diff(d):
    """perform order 1 diff only when the the values are strictly in/dec-cremental along the specified axis

    Note:
        skip those already taking a step like shape
        if we don't diff for steps, likely we don't need to handle spikes
        along the column
    """
    def step_level(dd):
        mid = int(dd.shape[0] / 2)
        dd = minmax(dd)
        left_dist = np.percentile(dd[:mid, :], [25, 50, 75], axis=0)
        right_dist = np.percentile(dd[mid:, :], [25, 50, 75], axis=0)
        # 25% - 75% take this as the step size;
        # when the two sides overlap badly, the step size would be even negative
        step_amp = np.maximum(right_dist[0, :] - left_dist[2, :],
                              left_dist[0, :] - right_dist[2, :])
        return step_amp


    d_diff = np.diff(d, axis=0)
    step_amp = step_level(d)

    nans = np.where(np.any(np.isnan(d_diff), axis=0))[0].tolist()
    if len(nans) > 0:
        print(f"adaptive_diff: Columns ({nans}) have NAN")

    # monotonous idx
    mono_idx = np.concatenate((np.where(np.all(d_diff >= 0, axis=0))[0],
                              np.where(np.all(d_diff <= 0, axis=0))[0]))

    # for monotonous columns at steady/even speed, step_amp ~ 0.125,
    # 0.5 is already a strong sign of step
    step_idx = np.where(step_amp > 0.5)[0]
    mono_idx = [i for i in mono_idx if i not in step_idx]
    d = d[1:, ]   # remove the first row to be inline with the diffed data
    d[:, mono_idx] = d_diff[:, mono_idx]
    return d

event_color = {'shutdown_interface': 'rgb(252, 110, 99)',
               'enable_interface': 'rgb(88, 188, 144)',
               'add_network_loop': 'rgb(190,174,212)',
               'remove_network_loop': 'rgb(102,194,165)',
               'rel_cpt': 'rgb(255, 127, 0)',
               'seed_cpt': 'rgb(190, 174, 212)'}

ignore_events = ['clear_telemetry', 'apply_device_configs', 'collect_telemetry']

def get_event_label(e):
    res = e['event']
    if e['device']:
        res += ' on ' + e['device']
    if e['interface']:
        res += ': ' + e['interface']
    return res

def get_event_color(e_str):
    """return the rgb color code of an event"""
    if e_str in event_color.keys():
        return event_color[e_str]
    elif e_str.startswith("gold"):
        return 'rgb(212,175,55)'
    else:
        return 'rgb(115,115,115)'

def plot_data_anime(data, tstp, events, color=None, symbol=None, show_events=True):
    if len(data.shape) > 2:
        print("Organize data in shape <n_sample, n_features>")
        return
    if data.shape[1] < 3:
        temp = np.zeros((data.shape[0], 3))
        temp[:, :data.shape[1]] = data
        data = temp
    dim_var = np.var(data, axis=0)
    dim_idx = np.argsort(dim_var)[::-1]
    data = data[:, dim_idx]
    
    rd_2d = dict(
            x=data[:, 0].tolist(),
            y=data[:, 1].tolist(),
            hovertext=[str(tstp[i]-tstp[0]) for i in range(len(tstp))],
            hoverinfo='text',
            mode='markers',
            marker=dict(
                symbol=symbol if not (symbol is None) else "circle",
                size=7,
                color=color if not (color is None) else "rgb(49, 130, 189)",
                colorscale='Viridis'
            ),
            line=dict(width=0.5, color='rgba(255,0,0,0.7)'),
            showlegend=False
        )
    
    rd_2d_trace = dict(
                x=data[:, 0].tolist(),
                y=data[:, 1].tolist(),
                mode='lines',
                line=dict(width=0.3, color='rgba(128,128,128,0.5)'),
                showlegend=False
            )
    text_anno = dict(
                x=data[[0, -1], 0].tolist(),
                y=data[[0, -1], 1].tolist(),
                mode='text',
                text=['Start', 'End'],
                textposition='bottom center',
                showlegend=False
    )
    plot_data = [dict(x=[], y=[], mode="markers", showlegend=False)]
    event_anchors = []
    
    for e in events:
        if e['event'] in ignore_events:
            continue
        anchor1 = np.argmin(np.abs(np.array(tstp)-float(e['timestamp'])))
        event_anchors.append(anchor1)
        
        if show_events:
            anchor2 = anchor1 + 3 if anchor1 < (len(tstp) - 4) else anchor1 - 1
            cut = data[anchor2, :2] - data[anchor1, :2]
            mid = (data[anchor2, :2] + data[anchor1, :2])/2
            cut = cut/np.linalg.norm(cut) * 6
            evname = get_event_label(e)
            plot_data.append({
                'mode': 'lines',
                'x': [mid[0] + cut[1], mid[0] - cut[1]],
                'y': [mid[1] - cut[0], mid[1] + cut[0]],
                'name': evname,
                'text': evname,
                'line': {
                    'color': get_event_color(e['event']),
                    'width': 5,
                },
                'showlegend': True,
                'legendgroup': "event_bars",
            })
            
    plot_data.extend([rd_2d, rd_2d_trace, text_anno])
    
    event_data = data[event_anchors,:]        
            
    event_rd_2d = dict(
            x=event_data[:, 0].tolist(),
            y=event_data[:, 1].tolist(),
            hovertext=[str(tstp[i]-tstp[0]) for i in event_anchors],
            hoverinfo='text',
            mode='markers',
            marker=dict(
                symbol=symbol if not (symbol is None) else "circle",
                size=7,
                color="rgb(255, 0, 0)",
                colorscale='Viridis'
            ),
            showlegend=False
        )
    plot_data.append(event_rd_2d)

    
    frames = [dict(data=[dict(x=[data[i, 0]],
                              y=[data[i, 1]],
                              mode='markers',
                              marker=dict(color='rgb(17,157,255)',
                                          line = dict(color = 'rgb(33, 240, 133)',width = 2),
                                          size=28, symbol='circle'))])
                              for i in list(range(data.shape[0]))[::max(int(data.shape[0]/400), 1)]]

    return plot_data, frames
