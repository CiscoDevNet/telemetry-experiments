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
import numpy as np
from .utils import minmax, diff, ft_dissect
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import re
from collections import defaultdict
import os

print(f'Module {__name__}, development version')

class YangDict:
    """given a word or a list of word,s estimate its protocol stack
    
    The estimation is done with a hardcoded word-to-layer dictionary saved in file
    """
    def __init__(self):
        self.d = {}
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'yang_cisco_xr_native_dict'), 'r') as fp:
            for line in fp.readlines():
                try:
                    if '#' in line:  # '#' preceds comments 
                        line = line.split('#')[0]
                    k, v = line.split(',')  
                    self.d[k] = float(v.strip())
                except ValueError:
                    pass
        print(f"loaded {len(self.d)} words")

    def word_query(self, k: str):
        return self.d.get(k, -1)

    def sentence_query(self, *args):
        # given a list of word, the network layer would be the min among them
        rt = 10
        for w in args:
            w_v = self.word_query(w)
            if w_v == -1:
                print(f"YangDict: unkown word <{w}>")
            rt = w_v if (w_v != -1) and (w_v < rt) else rt
        return rt

    def update(self, k: str, v: int):
        pass


def sort_metric_abs_step(d):
    """metric for sorting, the amplitude of stepwise change"""
    w = int(d.shape[0]/3)
    left = np.mean(d[:w, :], axis=0)
    # mid = np.mean(d[w:-w, :], axis=0)
    right = np.mean(d[-w:, :], axis=0)

    left_std = np.std(d[:w, :], axis=0)
    right_std = np.std(d[-w:, :], axis=0)

    step = abs(right-left)
    std = (left_std + right_std) / 2

    valid_steps = step > 4 * std
    return step * valid_steps


def sort_metric_rel_step(d):
    """metric for sorting, the amplitude of stepwise change, relativized by higher side, upbounded to 1"""
    w = int(d.shape[0]/3)
    left = np.mean(d[:w, :], axis=0)
    # mid = np.mean(d[w:-w, :], axis=0)
    right = np.mean(d[-w:, :], axis=0)
    return abs(right-left) / np.array(list(map(lambda s: 1 if s < 3 * np.finfo(float).eps else s, np.maximum(abs(left), abs(right)))))


def sort_metric_spike(d):
    """metric for sorting, the relative amplitude of a spike
    TODO: why geometric mean, why not simply arithmetic mean
    """
    w = int(d.shape[0]/3)
    left = np.mean(d[:w, :], axis=0)
    mid_h = np.max(d[w:-w, :], axis=0)
    mid_l = np.min(d[w:-w, :], axis=0)
    right = np.mean(d[-w:, :], axis=0)

    left_std = np.std(d[:w, :], axis=0)
    mid_std = np.std(d[w:-w, :], axis=0)
    right_std = np.std(d[-w:, :], axis=0)

    spike = np.array(mid_std * mid_std)
    base = np.array(left_std * right_std)

    valid_spikes = spike > 4 * base
    spike_amp = np.maximum(np.sqrt(abs(mid_h * mid_h - left*right)), np.sqrt(abs(mid_l * mid_l - left*right)))
    # spike_amp = np.maximum(np.maximum(np.maximum(abs(mid_h - left), abs(mid_h - right)), abs(mid_l - left)), abs(mid_l - right))

    return spike_amp * valid_spikes


def sort_metric_mix(d):
    """metric for sorting, combine step and spike amplitude"""
    return np.maximum(sort_metric_abs_step(d), sort_metric_spike(d))


def sort_metric_step_percentile(d, eve_pos=None, right_tol=1, left_tol=3):
    """step size is seen as the gap between the 25th pctl high end and 75th pctl low end
    
    Args:
        eve_pos (int or None): detection/event index to be explained
        right_tol (int): tolerence range befre eve_pos
        left_tol (int): tolerence range after eve_pos
    """
    if eve_pos is None:
        # take the center if not specified
        # eve_pos - left_tol > 0
        # eve_pos + right_tol < d.shape[0]
        eve_pos = int(d.shape[0] / 2)
    
    left_dist = np.percentile(d[:(eve_pos-left_tol), :], [25, 50, 75], axis=0)
    right_dist = np.percentile(d[(eve_pos+right_tol):, :], [25, 50, 75], axis=0)
    # 25th HIGH - 75th LOW take this as the step size;
    # when the two sides overlap badly, the step size would be even negative
    step_amp = np.maximum(right_dist[0, :] - left_dist[2, :],
                          left_dist[0, :] - right_dist[2, :])
    return step_amp


def sort_metric_spike_percentile(d, eve_pos=None, right_tol=1, left_tol=3):
    """spike size is seen as the gap between:
        * spike tip and 75th pctl high end
        * dip to and 25th pctl low end

        Args:
            eve_pos (int or None): detection/event index to be explained
            right_tol (int): tolerence range befre eve_pos
            left_tol (int): tolerence range after eve_pos
    """
    if eve_pos is None:
        # take the center if not specified
        # eve_pos - left_tol > 0
        # eve_pos + right_tol < d.shape[0]
        eve_pos = int(d.shape[0] / 2)

    left_dist = np.percentile(d[:(eve_pos-left_tol), :], [25, 50, 75], axis=0)
    right_dist = np.percentile(d[(eve_pos+right_tol):, :], [25, 50, 75], axis=0)

    mid_h = np.max(d[(eve_pos-left_tol):(eve_pos+right_tol), :], axis=0)
    mid_l = np.min(d[(eve_pos-left_tol):(eve_pos+right_tol), :], axis=0)
    # PEAK - 75th HIGH, or 25th LOW - DIP as the spike amp
    spike_amp = np.maximum(mid_h - np.maximum(left_dist[2,:], right_dist[2, :]),
                           np.minimum(left_dist[0, :], right_dist[0, :]) - mid_l)

    return spike_amp


def sort_metric_mix_percentile(d, eve_pos=None, right_tol=1, left_tol=3):
    """ metric for sorting, take stronger side of raw spike and raw step

    step size is seen as the gap between the 25th pctl high end and 75th pctl low end
    spike size is seen as the gap between:
    * spike tip and 75th pctl high end
    * dip to and 25th pctl low end

    the mix of step of spike will the larger one from them  

    By specifying the eve pos, we can achieve asymetric window.
    The calculation of pctl above shall live well with asymetric window
    [eve_pos - right_tol, eve_pos + left_tol) is where step and spike is supposed to happen,
    the min and max would correspond to the tip of spike/dip

    Args:
        eve_pos (int or None): detection/event index to be explained
        right_tol (int): tolerence range befre eve_pos
        left_tol (int): tolerence range after eve_pos
    """
    kargs = locals()
    return np.maximum(sort_metric_step_percentile(**kargs),
                      sort_metric_spike_percentile(**kargs))


def sort_metric_step_eyeQ(d, eve_pos=None, right_tol=1, left_tol=3):
    """The step size is calcuated as abs(mu_l - mu_r) / exp(sigma_l + sigmal_r)
    
    Args:
        eve_pos (int or None): detection/event index to be explained
        right_tol (int): tolerence range befre eve_pos
        left_tol (int): tolerence range after eve_pos
    """
    if eve_pos is None:
        # take the center if not specified
        # eve_pos - left_tol > 0
        # eve_pos + right_tol < d.shape[0]
        eve_pos = int(d.shape[0] / 2)

    left_mu = np.mean(d[:(eve_pos-left_tol), :], axis=0)
    right_mu = np.mean(d[(eve_pos+right_tol):, :], axis=0)
    left_sigma = np.std(d[:(eve_pos-left_tol), :], axis=0)
    right_sigma = np.std(d[(eve_pos+right_tol):, :], axis=0)

    # step_amp = np.abs(left_mu - right_mu) / np.exp(left_sigma + right_sigma)
    ran = np.max(d, axis=0) - np.min(d, axis=0)
    step_amp = np.abs(left_mu - right_mu) * (ran - (left_sigma + right_sigma))
    return step_amp


def sort_metric_spike_eyeQ(d, eve_pos=None, right_tol=1, left_tol=3):
    """The spike size is calculated as (spike - mu_high) / exp(sigma_high) or (dip - mu_low) /  exp(sigma_low)
    [eve_pos - right_tol, eve_pos + left_tol) is where step and spike is supposed to happen
    The min and max would correspond to the tip of spike/dip"""
    if eve_pos is None:
        # take the center if not specified
        # eve_pos - left_tol > 0
        # eve_pos + right_tol < d.shape[0]
        eve_pos = int(d.shape[0] / 2)

    left_mu = np.mean(d[:(eve_pos-left_tol), :], axis=0)
    right_mu = np.mean(d[(eve_pos+right_tol):, :], axis=0)
    left_sigma = np.std(d[:(eve_pos-left_tol), :], axis=0)
    right_sigma = np.std(d[(eve_pos+right_tol):, :], axis=0)

    high_mu = np.maximum(left_mu, right_mu)
    low_mu = np.minimum(left_mu, right_mu)
    
    high_sigma = np.empty(left_sigma.size, dtype=left_sigma.dtype)
    high_sigma[np.where(left_mu > right_mu)[0]] = left_sigma[np.where(left_mu > right_mu)[0]]
    high_sigma[np.where(left_mu <= right_mu)[0]] = right_sigma[np.where(left_mu <= right_mu)[0]]

    low_sigma = np.empty(left_sigma.size, dtype=left_sigma.dtype)
    low_sigma[np.where(left_mu > right_mu)[0]] = right_sigma[np.where(left_mu > right_mu)[0]]
    low_sigma[np.where(left_mu <= right_mu)[0]] = left_sigma[np.where(left_mu <= right_mu)[0]]
    
    mid_h = np.max(d[(eve_pos-left_tol):(eve_pos+right_tol), :], axis=0)
    mid_l = np.min(d[(eve_pos-left_tol):(eve_pos+right_tol), :], axis=0)
    
    # randomly run into numerical error here with np.exp
    # spike_amp = np.maximum((mid_h - high_mu) / np.exp(high_sigma),
    #                       (low_mu - mid_l) / np.exp(low_sigma))
    
    spike_amp = np.maximum((mid_h - high_mu) - high_sigma,
                           (low_mu - mid_l) - low_sigma )
    return spike_amp


def sort_metric_mix_eyeQ(d, eve_pos=None, right_tol=1, left_tol=3):
    """metric for sorting, take stronger side of raw spike and raw step
    
    the calculation takes some idea from the eye pattern and the Q-factor in telecommunication receiver
    https://mapyourtech.com/entries/general/what-is-q-factor-and-what-is-its-importance-

    (Gaussiance assumption; sample numbers too small in our case.)
    
    The step size is calcuated as abs(mu_l - mu_r) / exp(sigma_l + sigmal_r)
    The spike size is calculated as (spike - mu_high) / exp(sigma_high) or (dip - mu_low) /  exp(sigma_low)
    [eve_pos - right_tol, eve_pos + left_tol) is where step and spike is supposed to happen
    The min and max would correspond to the tip of spike/dip

    The Q-factor of an eye pattern is calculated as (mu_1 - mu_2) / (sigma_1 + sigma_2)
    We are not using it directly because that (sigma_1 + sigma_2) could be 0 in our case with perfect step/spike
    Need a way to perserve the full abs(mu_l - mu_r) strength, when the sigma sum is 0
    Also, as the sigma sum increases, the step/spike amplitude shall be penalized/shrinked from the orignal full abs(mu_l - mu_r) strength.
    exp(sigma_l + sigmal_r) is 1 when the sigma sum is 0, and is an increasing func to sigma sum.
    
    One issue is rather whether the measures are faire across spike and step, as spik is penalized with one sided sigma

    Args:
        eve_pos (int or None): detection/event index to be explained
        right_tol (int): tolerence range befre eve_pos
        left_tol (int): tolerence range after eve_pos
    """
    kargs = locals()
    return np.maximum(sort_metric_step_eyeQ(**kargs),
                      sort_metric_spike_eyeQ(**kargs))


def sort_metric_step_2in1(d, eve_pos=None, right_tol=1, left_tol=3):
    """calcualte the step size for raw, diffed and cusum, pick the largest"""
    raw_step = sort_metric_step_eyeQ(minmax(d), eve_pos, right_tol, left_tol)
    diff_step = sort_metric_step_eyeQ(minmax(np.diff(d, axis=0)), eve_pos, right_tol, left_tol)

    return np.max(np.array([raw_step, diff_step]), axis=0)


def sort_metric_var(d):
    """metric for sorting, standard deviation"""
    return np.std(d, axis=0)


def sort_metric_ref_dis(d, win=1):
    """metric for sorting, min DTW distance to reference time series
    TODO: accelerate by skipping low variance ones
    """
    w = d.shape[1]
    l = d.shape[0]
    # reference shape
    ref = [
        np.array([(0 if i < l / 2 else 1) for i in range(l)]),  # __|--
        np.array([(1 if i < l / 2 else 0) for i in range(l)]),  # --|__
        np.array([(0 if i != int(l / 2) + 1 else 1) for i in range(l)]),  #  __|__
        np.array([(1 if i != int(l / 2) + 1 else 0) for i in range(l)])  # --|--
        ]
    # anti reference shape
    anti_ref = [
        np.array([0 for _ in range(l)]),
        np.array([1 for _ in range(l)]),
        minmax(np.random.normal(size=l))]
    d_var = np.std(d, axis=0)
    # the max distance is l
    # if std is 0, set dis_to_ref to max, dis_to_anti_ref to 0
    # w * ref_num as resulted dimension
    # fastdtw is not selective enough
    # euclidean dis is not friendly with spikes
    dis_to_ref = np.array([[l if d_var[j] == 0 else np.linalg.norm(i-d[:,j].T) for i in ref] for j in range(w)])
    dis_to_anti_ref = np.array([[(0 if d_var[j] == 0 else np.linalg.norm(i-d[:,j].T)) for i in anti_ref] for j in range(w)])
    # take the smallest for each feature
    r = np.min(dis_to_ref.T, axis=0)
    p = np.min(dis_to_anti_ref.T, axis=0)
    # if a feature closer to anti -> dis_to_anti, else l-dis_to_ref, normalize to [0, 1], larger the better
    return  (l - np.array(list(map(lambda s: s[0] if s[0] < s[1] else l-s[1], zip(r, p))))) / float(l)

def pca_metric(d):
    d = minmax(diff(d))
    m = np.mean(d, axis=0)
    d = (d - m)
    pca = PCA(n_components=3).fit(d)
    print("PCA explained variance: %.2f%%" % (100*sum(pca.explained_variance_ratio_)))
    res = pca.transform(np.eye(d.shape[1]))
    contrib = np.sum(np.abs(res), axis=1)
    return contrib / max(contrib)

def lda_metric(d):
    d = diff(d)
    m = np.mean(d, axis=0)
    d = (d - m)
    mid = int(d.shape[0]/2)
    y = np.concatenate((np.zeros(mid), np.ones(d.shape[0]-mid)))
    lda = LinearDiscriminantAnalysis().fit(d, y)
    res = lda.transform(np.eye(d.shape[1]))
    contrib = np.sum(np.abs(res), axis=1)
    return contrib / max(contrib)

def shortlist_by_cliff(v):
    """stopping criterion, before the larggest cliff"""
    return np.argmax(abs(np.diff(v)))


def tf_idf(fdist, ft_names, num_tokens):
    numofdocs= num_sensorpathwith_token(ft_names, fdist)
    totalftnum = len(ft_names)
    idf, tf ,tf_idf= [], [], []
    for i in range(len(fdist)):
        tf.append((fdist[i][0],fdist[i][1]/num_tokens))
        idf.append((fdist[i][0], np.log(totalftnum/numofdocs[i][1])))
        tf_idf.append((fdist[i][0], idf[i][1]*tf[i][1]))
    return tf_idf

def num_sensorpathwith_token(ft_names, fdist):
    num_output = []
    for i in range(len(fdist)):
        counter = 0
        for ft in ft_names:
            if fdist[i][0] in re.split('\_|:|\.', ft):
                counter+=1
        num_output.append((fdist[i][0], counter))
    return num_output

def summary(fts, scores, scores_sub, kv_norm=None):
    """text summeraization on shortlisted features and their scores

    summerize via grouping by kv, all encoding-paths mixed up
    """
    kv_set = defaultdict(float)  # for accumulated score
    kv_set_sub = defaultdict(float)
    kv_lf_set = defaultdict(set)
    kv_md_set = defaultdict(set)
    kv_set_idx = defaultdict(list)

    if kv_norm is None:
        kv_norm = defaultdict(int)  # base number, norm of each kv

    for i, (ft, sc, sc_sub) in enumerate(zip(fts, scores, scores_sub)):
        ymd, sp, kv, leaf = ft_dissect(ft)
        kv_str = ':'.join(kv)
        kv_set[kv_str] += sc  # accumulate the score at the presence of feature
        kv_set_sub[kv_str] = max(kv_set_sub[kv_str], sc_sub)
        kv_set_idx[kv_str].append(i)
        kv_norm[kv_str] += 1
        kv_lf_set[kv_str].add(leaf)  # only unique leaf is taken
        kv_md_set[kv_str].add(ymd)

    # the order is by kv_set[k]/kv_norm[k], decremental
    # return ["<br>".join(pprint.pformat({k: {'score': v, 'base':kv_norm[k], 'modules': kv_md_set[k], 'leafs': kv_lf_set[k]}}, width=50).split("\n"))
    #         for k, v in sorted(kv_set.items(), key=lambda s: s[1] / float(kv_norm[s[0]]), reverse=True) if kv_set_sub[k] > 0.1]

    return [(k, {'score': v, 'base':kv_norm[k], 'aux_score':kv_set_sub[k], 'modules': kv_md_set[k], 'leafs': kv_lf_set[k], 'idx': kv_set_idx[k]})
            for k, v in sorted(kv_set.items(), key=lambda s: s[1] / float(kv_norm[s[0]]), reverse=True) if kv_set_sub[k] > 0.1]


def summary_cross(fts, scores):
    """text summeraization on shortlisted features and their scores

    cross in the sense it decides whether group by kv or by leaf, on a per encoding-path/sensor-path base
    """
    ep_set = dict()

    for i, (ft, sc) in enumerate(zip(fts, scores)):
        ymd, sp, kv, leaf = ft_dissect(ft)
        kv_str = ':'.join(kv)
        ep = ':'.join([ymd, sp])  # encoding path is the concatenation of module name and sensor-path underneath
        if ep not in ep_set:
            # for each ep, we count the accumlated score by kv and by leaf
            ep_set[ep] = {'by_kv':{'cum_score': defaultdict(int), 'leafs': defaultdict(list), 'idx': defaultdict(list)},
                      'by_leaf':{'cum_score':defaultdict(int), 'kvs': defaultdict(list), 'idx': defaultdict(list)}}
        ep_set[ep]['by_kv']['cum_score'][kv_str] += sc
        ep_set[ep]['by_kv']['leafs'][kv_str].append(leaf)
        ep_set[ep]['by_kv']['idx'][kv_str].append(i)
        ep_set[ep]['by_leaf']['cum_score'][leaf] += sc
        ep_set[ep]['by_leaf']['kvs'][leaf].append(kv_str)
        ep_set[ep]['by_leaf']['idx'][leaf].append(i)

    final_sort = {}  # the dict used in final sort across eps
    for ep in ep_set:
        # the vote for "group by kv"/"group by leaf" equals the max accumlated score assciated to a kv or leaf of the encoding path
        # if group by kv gives larger max cum_score, the entire encoding path is summarized in kv groups
        # TODO: this criterion to be revisited
        kv_vote = max(ep_set[ep]['by_kv']['cum_score'].values())
        lf_vote = max(ep_set[ep]['by_leaf']['cum_score'].values())
        if kv_vote > lf_vote:
            for kv in ep_set[ep]['by_kv']['cum_score']:
                final_sort[ep+"|"+kv] = {"cum_score": ep_set[ep]['by_kv']['cum_score'][kv],
                                         "leafs": set(ep_set[ep]['by_kv']['leafs'][kv]),
                                         "base" : len(ep_set[ep]['by_kv']['leafs'][kv]),  # TODO: is it possible having two same leaves within an ep?
                                         "encoding-path": ep,
                                         "idx": ep_set[ep]['by_kv']['idx'][kv]}
        else:
            for lf in ep_set[ep]['by_leaf']['cum_score']:
                final_sort[ep+"|"+lf] = {"cum_score": ep_set[ep]['by_leaf']['cum_score'][lf],
                                         "key-values": set(ep_set[ep]['by_leaf']['kvs'][lf]),
                                         "base": len(ep_set[ep]['by_leaf']['kvs'][lf]),
                                         "encoding-path": ep,
                                         "idx": ep_set[ep]['by_leaf']['idx'][lf]}

    # return [(k.split("|")[-1], v)
    #        for k, v in sorted(final_sort.items(), key=lambda s: s[1]["cum_score"]/float(s[1]["base"]),
    #        reverse=True) if (v["cum_score"] / float(v["base"])) > 0.85]
    
    return [(k.split("|")[-1], v)
           for k, v in sorted(final_sort.items(), key=lambda s: s[1]["cum_score"]/float(s[1]["base"]),
           reverse=True)]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_ratio(d, scale=False):
    """cross event angle / pre event angle
    signal vs noise
    """
    w = int(d.shape[0]/3)
    pre_event = d[:w, :]
    post_event = d[-w:, :]
    if scale:
        step = np.maximum(np.mean(pre_event, axis=0), 1e6)
        pre_event = pre_event / step
        post_event = post_event / step
    pre_event_angles = [angle_between(pre_event[i-1, :], pre_event[i,:]) for i in range(1, w)]
    # print(f"Pre-event angles: {pre_event_angles}")
    pre_event_mean = np.mean(pre_event, axis=0)
    post_event_mean = np.mean(post_event, axis=0)
    cross_event_angle = angle_between(pre_event_mean, post_event_mean)
    # print(f"Cross-event angle: {cross_event_angle}")
    return {"angle ratio": abs(cross_event_angle - np.mean(pre_event_angles)) / np.std(pre_event_angles),
            "sin-square ratio": abs(np.square(np.sin(cross_event_angle)) - np.square(np.sin(np.mean(pre_event_angles)))) /np.std(np.square(np.sin(pre_event_angles)))}


def shfit_over_std(d, scale=False):
    """ distance of shift over data std of the pre event status
    """
    w = int(d.shape[0]/3)
    pre_event = d[:w, :]
    post_event = d[-w:, :]
    if scale:
        step = np.maximum(np.mean(pre_event, axis=0), 1e6)
        pre_event = pre_event / step
        post_event = post_event / step
    ma = np.mean(pre_event, axis=0)  # pre event center
    mb = np.mean(post_event, axis=0)  # post event center
    shift = mb - ma  # shift vector
    u_shift = unit_vector(shift)  # unite vector on the shift direction
    pre_event_std = np.std(np.dot((pre_event - ma), u_shift.T ))  # project pre event data on to shift direction and then calculate std
    return {"shift distance": np.linalg.norm(shift),
            "pre event std": pre_event_std,
            "ratio": np.linalg.norm(shift)/pre_event_std}


if __name__ == '__main__':
    """some test code on sort metric and adaptive diff"""
    from data_utils import adaptive_diff, minmax
    from plotly.offline import plot
    import plotly.graph_objects as go

    def plot_test_data(data, name, text=None, fn=None):
        fig = go.Figure()
        for i in range(data.shape[0]):
            fig.add_trace(dict(type='scatter', x=list(range(data.shape[1])), y=data[i, :],
                            mode='markers+lines', name=name[i], text=text[i] if text is not None else None))
        fig.add_shape(dict(type='line', 
                           yref='paper', y0= 0, y1=1,
                           xref= 'x', x0=(data.shape[1])/2 - 3, x1=(data.shape[1])/2 - 3, ))
        fig.add_shape(dict(type='line', 
                           yref='paper', y0= 0, y1=1,
                           xref= 'x', x0=(data.shape[1])/2 + 1, x1=(data.shape[1])/2 + 1, ))
        plot(fig, auto_open=True, filename=fn if fn is not None else "temp-plot.html")

    test_data = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # flat
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],  # oscillating
        [3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3],  # less frequenct oscillation
        [1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1],  # Thomas oscillation
        [9, 9, 8, 9, 9, 8, 9, 9, 0, 9, 6, 6, 5, 5, 6, 6, 5],  # down going spike
        [3, 2, 1, 2, 3, 4, 2, 4, 10, 0, 4, 3, 2, 2, 1, 2, 1],  # spike with oscilation
        [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Thomas spike
        [1, 2, 2, 2, 2, 3, 3, 3, 4, 13, 14, 14, 14, 14, 15, 15, 16],  # step in trend
        [1, 2, 3, 2, 1, 3, 3, 3, 4, 13, 1, 14, 1, 18, 3, 14, 0],  # step with one side oscillation
        [1, 2, 1, 0, 2, 1, 1, 2, 6, 7, 8, 6, 5, 5, 0, 5, 6],  # step with oscilation
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # perfect step
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # perfect spike
        ])
    test_data_name = ["flat",
                      "oscillating", "less frequenct osciallation", "Thomas oscillation",
                      "down going spike", "spike with oscilation", "Thomas spike",
                      "step in trend", "step with one side oscillation", "step with oscilation",
                      "perfect step",  "perfect spike"]
    
    pre_proc_data = adaptive_diff(minmax(test_data.T))

    std_step = sort_metric_abs_step(pre_proc_data)
    std_spike = sort_metric_spike(pre_proc_data)


    percentile_step = sort_metric_step_percentile(pre_proc_data)
    percentile_spike = sort_metric_spike_percentile(pre_proc_data)

    eyeQ_step = sort_metric_step_eyeQ(pre_proc_data)
    eyeQ_spike = sort_metric_spike_eyeQ(pre_proc_data)

    sort_metric_res = [f"{test_data_name[i]}, metric type (step amp, spike amp):\
                        std({std_step[i]:.3f}, {std_spike[i]:.3f}),\
                        pctile({percentile_step[i]:.3f}, {percentile_spike[i]:.3f}),\
                        eyeQ({eyeQ_step[i]:.3f}, {eyeQ_spike[i]:.3f})" 
                        for i in range(len(test_data_name))]

    plot_test_data(test_data, test_data_name, fn='raw_data.html')
    plot_test_data(pre_proc_data.T, test_data_name, sort_metric_res, fn='minmax_diff_sort_res.html')
