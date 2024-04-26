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

from collections import deque

import numpy as np

from IPython.display import display, Markdown, HTML
import ipywidgets as widgets


def minmax(d, axis=0):
    """"min-max scaler"""
    ran = np.asarray(np.max(d, axis=axis) - np.min(d, axis=axis)).reshape(-1)
    zero_idx = np.argwhere(ran == 0)
    ran[zero_idx] = 1
    return (d-np.min(d, axis=axis)) / ran

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

ITEM_LAYOUT = widgets.Layout(display="flex", justify_content="flex-start", width="max-content")
ITEM_STYLE = {'description_width': '200px'}

def jupyter_display_options(*items):
    display(
        widgets.Box(items, layout=widgets.Layout(flex_flow='column nowrap'))
    )

def jupyter_text(description, value):
    return widgets.Text(
        description=description, 
        value=value, 
        layout=ITEM_LAYOUT, 
        style=ITEM_STYLE)

def jupyter_dropdown(description, options, value):
    return widgets.Dropdown(
        description=description,
        options=options,
        value=value,
        disabled=False,
        layout=ITEM_LAYOUT,
        style=ITEM_STYLE
    )