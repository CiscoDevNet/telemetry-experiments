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
from collections import deque
from .utils import ft_dissect

print(f'Module {__name__}, development version')


# --------- Feature Name Processing ---------

hex_list = []
for i in range(256):
    t = str(hex(i))[2:]
    if len(t) < 2:
        t = '0'+ t
    hex_list += [t]

def _is_ip(s):
    return all(e.isdigit() and int(e)<256 for e in s.split('.')) and len(s.split('.')) == 4

def _ft_dissect(ft_string):
    """dissect a feature name into module name, key value array, and leaf name"""
    parsed = ft_dissect(ft_string)
    return parsed[1], ' '.join(parsed[2]), parsed[3]


# --------------- Metrics ----------------

def occurence(names_set):
    """ Computes normalized occurences of string in set """
    names, counts = np.unique(names_set, return_counts=True)
    proba = counts/sum(counts)
    occurence = {}
    for n, p in zip(names, proba):
        occurence[n] = p
    return occurence

def distribution(dissected_features):
    """ Computes normalized occurences of sensor paths """
    mod_occurence = occurence([d[0] for d in dissected_features])
    kv_occurence = occurence([d[1] for d in dissected_features])
    leaf_occurence = occurence([d[2] for d in dissected_features])
    return mod_occurence, kv_occurence, leaf_occurence

def entropy(selected_occurence):
    """ Entropy """
    ent = 0
    for (_, p) in selected_occurence.items():
        ent += - p * np.log(p)
    return ent

def cross_entropy(full_occurence, selected_occurence):
    """ Cross Entropy H(p,q) """
    x_ent = 0
    for (name, p) in selected_occurence.items():
        q = full_occurence[name]
        x_ent += - p * np.log(q)
    return x_ent

def cross_entropy_gain(full_occurence, selected_occurence):
    """ Cross Entropy gain G = H(p,q) - H(q) """
    x_ent = 0
    for (name, q) in full_occurence.items():
        if name in selected_occurence.keys():
            p = selected_occurence[name]
            x_ent += (q - p) * np.log(q)
        else:
            x_ent += q * np.log(q)
    return x_ent

def score(ft_store, full_dist, selected_names_idx):
    """ Computes score of selection from a full dataset """
    if len(selected_names_idx) == 0:
        return 0
    selected_dissected = [[ft_store.get_encpath(fn),
                           ft_store.get_joined_kv(fn),
                           ft_store.get_joined_path(fn)] for fn in selected_names_idx]
    selected_dist = distribution(selected_dissected)

    mod_xent = cross_entropy(full_dist[0], selected_dist[0])
    kv_xent = cross_entropy(full_dist[1], selected_dist[1])
    leaf_xent = cross_entropy(full_dist[2], selected_dist[2])

    return sum([mod_xent, kv_xent, leaf_xent])


# ------------------- Main ------------------

def run_opti(ft_store, full_names_idx, scores, alpha=2, N_max_epochs=20):
    if len(scores) == 0:
        return []

    # Dissect and compute the distribution over the feature names
    full_dissected = [[ft_store.get_encpath(fn), ft_store.get_joined_kv(fn), ft_store.get_joined_path(fn)] for fn in full_names_idx]
    full_dist = distribution(full_dissected)

    args = np.argsort(scores)[::-1]
    sorted_scores = scores[args]
    N_total = min(max(np.argmax(abs(np.diff(sorted_scores))), 1), 1000)
    N_total = min(N_total, scores.shape[0])
    sorted_names = [full_names_idx[a] for a in args]
    sorted_names = sorted_names[:N_total]
    sorted_scores = sorted_scores[:N_total]
    change_count = 1
    selected_idx = list(range(max(int(N_total/2), 1)))

    print("\nRunning optimisation...")
    print('-' * 50)
    print("Total features considered: %s\nAlpha: %s" % (N_total, alpha))
    print('-' * 50)
    epoch = 0

    N_ref = len(selected_idx)
    selected_names = [sorted_names[s] for s in selected_idx]
    cross_ent = score(ft_store, full_dist, selected_names)
    change_score = np.mean(sorted_scores[selected_idx])
    ref = (1 - np.exp(- N_ref / alpha)) * cross_ent * change_score

    while change_count > 0 and epoch < N_max_epochs:
        i_ref = ref
        i_selected_idx = selected_idx

        epoch += 1
        change_count = 0

        to_remove = []
        for (k, i) in enumerate(selected_idx):
            new_selected_idx = selected_idx.copy()
            new_selected_idx.remove(i)
            new_names = [sorted_names[nsi] for nsi in new_selected_idx]
            cross_ent = score(ft_store, full_dist, new_names)
            change_score = np.mean(sorted_scores[new_selected_idx])
            d = (1 - np.exp(- (N_ref - 1) / alpha)) * cross_ent * change_score
            if d > ref:
                to_remove.append(i)
                change_count += 1
        
        if len(to_remove) == len(selected_idx):
            to_remove.remove(to_remove[0])
        selected_idx = [idx for idx in selected_idx if idx not in to_remove]

        N_ref = len(selected_idx)
        selected_names = [sorted_names[si] for si in selected_idx]
        cross_ent = score(ft_store, full_dist, selected_names)
        change_score = np.mean(sorted_scores[selected_idx])
        mid_ref = (1 - np.exp(- N_ref / alpha)) * cross_ent * change_score
        mid_selected_idx = selected_idx

        non_selected_idx = [idx for idx in range(N_total) if idx not in selected_idx]
        to_add = []
        for i in non_selected_idx:
            new_selected_idx = selected_idx.copy()
            new_selected_idx.append(i)
            new_names = [sorted_names[nsi] for nsi in new_selected_idx]
            cross_ent = score(ft_store, full_dist, new_names)
            change_score = np.mean(sorted_scores[new_selected_idx])
            d = (1 - np.exp(- (N_ref + 1) / alpha)) * cross_ent * change_score
            if d > mid_ref:
                to_add.append(i)
                if i not in to_remove:
                    change_count += 1
                else:
                    change_count -= 1
        selected_idx += to_add

        N_ref = len(selected_idx)
        selected_names = [sorted_names[si] for si in selected_idx]
        cross_ent = score(ft_store, full_dist, selected_names)
        change_score = np.mean(sorted_scores[selected_idx])
        ref = (1 - np.exp(- N_ref / alpha)) * cross_ent * change_score

        if mid_ref > ref:
            selected_idx = mid_selected_idx
            ref = mid_ref
        if i_ref > ref:
            selected_idx = i_selected_idx
            ref = i_ref
            change_count = 0

        print("Epoch %s : Score = %.2f" % (epoch, ref))

    score_gains = []
    for i in selected_idx:
        new_selected_idx = selected_idx.copy()
        new_selected_idx.remove(i)
        new_names = [sorted_names[nsi] for nsi in new_selected_idx]
        cross_ent = score(ft_store, full_dist, new_names)
        change_score = np.mean(sorted_scores[new_selected_idx])
        score_gains.append((1 - np.exp(- (N_ref - 1) / alpha)) * cross_ent * change_score)

    gain_s_args = np.argsort(score_gains)
    selected_idx = np.array(selected_idx)[gain_s_args]

    return [sorted_names[si] for si in selected_idx]
