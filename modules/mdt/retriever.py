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

#!/usr/bin/env python3

import os
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

from .traffic_leaf_classifier import traffic_leaf_test
from . import explain_lib as explib
from .traffic_leaf_classifier import traffic_leaf_test
from .selection_lib import run_opti
from .feature_store import FeatureStore
from .utils import minmax, adaptive_diff, ft_dissect

class Retriever:

    def __init__(self, data : pd.DataFrame):
        self.dataframe = data

    def retrieve(self, changepoints, one_sided_window: int = 150, max_selection_size: int = 5,
                 regularization_term: int = 2, max_number_of_epochs: int = 20 ):

        traffic_metric_name = 'sort_metric_step_eyeQ'

        # get metric func
        metric_func = getattr(explib, traffic_metric_name)

        fulldata = self.dataframe.to_numpy(dtype=float)
        tstp = fulldata[:,0]
        data = fulldata[:,1:]        
        ft_names = self.dataframe.columns.values[1:]
        
        ft_store = FeatureStore(ft_names)
        ft_names_idx = list(range(len(ft_names)))
        
        ft_class = []  # type of each feature
        # leaf count associate to each kv
        mainft_leaf_cnt = defaultdict(int)
        traffic_leaf_cnt = defaultdict(int)

        onbox_name_format = '[' in ft_store.get_flat_name(0)

        for fti in ft_names_idx:
            if onbox_name_format:
                leaf = ft_store.get_joined_path(fti)
                kv_str = ft_store.get_joined_kv(fti)
            else:
                _, _, kv, leaf = ft_dissect(ft_store.get_flat_name(fti))
                kv_str = ':'.join(kv)

            if traffic_leaf_test(leaf):
                ft_class.append(1)
                traffic_leaf_cnt[kv_str] += 1
            else:
                ft_class.append(-1)
                mainft_leaf_cnt[kv_str] += 1

        ft_class = np.array(ft_class)
        main_ft_idx = np.where(ft_class == -1)[0]
        traffic_ft_idx = np.where(ft_class == 1)[0]
        
        all_features = {}
        for timestamp in changepoints:
            
            # area of interest
            win_idx = np.where(abs(tstp - timestamp) <= one_sided_window)[0]
            # in window data preprocessing, take care of the numpy view and copy in slicing the data
            cp_window = data[win_idx,:]
            data_slice_shape = minmax(adaptive_diff(cp_window))  # cares only about the shape
            data_slice_raw = adaptive_diff(cp_window)

            # sorting metrics on a per counter/feature base
            main_metric = metric_func(data_slice_shape[:, main_ft_idx])
            traffic_metric = metric_func(data_slice_raw[:, traffic_ft_idx])

            sorted_main = np.array(sorted(enumerate(main_metric.tolist()), key=lambda s: s[1], reverse=True))
            sorted_traffic = np.array(sorted(enumerate(traffic_metric.tolist()), key=lambda s: s[1], reverse=True))

            if len(sorted_traffic) > 0:
                traffic_ft_names = [ft_names_idx[i] for i in traffic_ft_idx]
                t_names = [traffic_ft_names[i] for i in sorted_traffic[:,0].astype(int)]
                traffic_scores = sorted_traffic[:,1] / max(sorted_traffic[:,1])
            else:
                t_names = []
                traffic_scores = np.array([])

            if len(sorted_main) > 0:
                main_ft_names = [ft_names_idx[i] for i in main_ft_idx]
                m_names = [main_ft_names[i] for i in sorted_main[:,0].astype(int)]
                main_scores = sorted_main[:,1]
            else:
                m_names = []
                main_scores = np.array([])

            full_names = m_names + t_names
            scores = np.concatenate((main_scores, traffic_scores))
            
            selection = run_opti(ft_store, full_names, scores, alpha=regularization_term, 
                                 N_max_epochs=max_number_of_epochs)
            
            features = []
            for x in range(0, min(len(selection), max_selection_size)):
                cp_feature = ':'.join(ft_store.get_flat_name(selection[x]).split(':')[1:]) + " CHANGE: " 
                cp_feature +=  str(cp_window[:, selection[x]][-1] - cp_window[:, selection[x]][0])
                features.append(cp_feature)

            all_features[timestamp] = features

        return all_features