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

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from modules.mdt.data_utils import plot_data_anime
import plotly.graph_objects as go
import modules.mdt.utils as utils


class ChangepointDetector:

    def __init__(self, data : pd.DataFrame, device):
        self.dataframe = data
        self.device = device
        self.changes = None
        self.reduced = None
        self.tstp = None

    def detect(self, max_data_point_distance = 0.05):
    
        fulldata = self.dataframe.to_numpy(dtype=float)
        self.tstp = fulldata[:,0]
        data = fulldata[:,1:]

        solver = TSNE(n_components=2, init='pca', random_state=0)
        self.reduced = solver.fit_transform(data)
        
        solver = DBSCAN(eps = max_data_point_distance)
        clusters = solver.fit(MinMaxScaler().fit_transform(self.reduced)).labels_

        self.changes = np.where(clusters[:-1] != clusters[1:])[0]
        self.selectable = ["ALL"]
        self.events = []
        self.changepoints = []
        for i, t in enumerate(self.changes):
            timestamp = self.tstp[t]
            event = str(i+1)
            selector = event + " " + self.device
            
            self.events.append(
            {
                "timestamp": timestamp,
                "event": event,
                "device": self.device,
                "selector": selector,
                "interface": None   
            })
            self.selectable.append(selector)
            self.changepoints.append(timestamp)

        return self.changepoints
    
    def get_changepoints(self):
        
        if self._changepoints_dropbox is not None and self._changepoints_dropbox.value != "ALL":
            for event in self.events:
                if event["selector"] == self._changepoints_dropbox.value:
                    return [event["timestamp"]]
        
        return self.changepoints
    
    def select_changepoints(self):
        
        self._changepoints_dropbox = utils.jupyter_dropdown(
            description = "Changepoint Selection:",
            options = self.selectable,
            value = self.selectable[0]
        )
        
        utils.jupyter_display_options(self._changepoints_dropbox)
                    
    def plot(self, withEvents=True):
                
        if self.events is not None:
            plot_data, frames = plot_data_anime(self.reduced, self.tstp, self.events, color='rgb(128,177,211)', show_events=withEvents)
        else:
            plot_data, frames = plot_data_anime(self.reduced, self.tstp, [], color='rgb(128,177,211)', show_events=False)
            
        fig = go.Figure(
                data = plot_data,
                layout = {
                    'title': "tSNE 2-D Visualization",
                    'autosize': False,
                    'width': 1000,
                    'height': 1000,
                    'updatemenus': [{
                        'buttons': [
                            {
                                'args': [None, {
                                    'frame': {'duration': 100, 'redraw': False},
                                    'fromcurrent': True, 'transition': {'duration': 50, 'easing': 'quadratic-in-out'}}],
                                    'label': 'Go', 'method': 'animate'
                            },
                            {
                                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                'transition': {'duration': 0}}],
                                'label': 'Pause',
                                'method': 'animate'
                            }],
                        'direction': 'left',
                        'pad': {'r': 10, 't': 10},
                        'showactive': False,
                        'type': 'buttons',
                        'x': 0.1,
                        'xanchor': 'right',
                        'y': 1,
                        'yanchor': 'bottom'
                    }]},
                frames = frames)

        fig.show()
        