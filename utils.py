# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import numpy as np


def group_resuts(labels, preds, users):
    pass
    """Devide labels and preds into several group according to values in group keys.
    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.
    Returns:
        all_labels: labels after group.
        all_preds: preds after group.
    """

    all_keys = list(set(users))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for l, p, k in zip(labels, preds, users):
        group_labels[k].append(l)
        group_preds[k].append(p)

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return all_labels, all_preds


def mrr_score(y_true, y_score):
    """Computing mrr score metric.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: hit score.
    """
    ground_truth = np.where(np.array(y_true) == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def p_score(y_true, y_score, k=1):
    """Computing precision score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: precision scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    p_score = np.sum(y_true[..., :k], -1)
    return np.mean(p_score)

'''
def p_score(y_true, y_score, k=1):
    """Computing precision score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: precision scores.
    """
    ipdb.set_trace()
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    p_score = np.sum(y_true[..., :k], -1)
    return np.mean(p_score)
'''

def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.

    FIXME:
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        elif metric.startswith("P"):    # format like:  P@1;3
            k_list = [1]
            for k in k_list:
                p_k = np.mean(
                    [
                        p_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["P@{0}".format(k)] = round(p_k, 4)
        else:
            raise ValueError("not define this metric {}".format(metric))
    return res
