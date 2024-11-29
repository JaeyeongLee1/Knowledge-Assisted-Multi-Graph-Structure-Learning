import numpy as np
from util.data import *


def get_full_err_scores(test_result, val_result, epsilon, b_num):
    test_predict, test_gt, _ = test_result
    val_predict, val_gt, _ = val_result

    test_predict_np = np.array(test_predict)
    test_gt_np = np.array(test_gt)

    val_predict_np = np.array(val_predict)
    val_gt_np = np.array(val_gt)

    test_delta = np.abs(np.subtract(test_predict_np, test_gt_np))
    val_delta = np.abs(np.subtract(val_predict_np, val_gt_np))

    n_err_mid, n_err_iqr = np.median(test_delta, axis=0), iqr(test_delta, axis=0)
    n_err_mid_val, n_err_iqr_val = np.median(val_delta, axis=0), iqr(val_delta, axis=0)

    norm_test_delta = (test_delta - n_err_mid) / (n_err_iqr + epsilon)
    norm_val_delta = (val_delta - n_err_mid_val) / (n_err_iqr_val + epsilon)

    norm_test_delta_sm = np.empty_like(norm_test_delta)
    norm_val_delta_sm = np.empty_like(norm_val_delta)
    assert norm_test_delta_sm.shape[-1] == norm_val_delta_sm.shape[-1]

    for i in range(norm_test_delta_sm.shape[-1]):
        norm_test_delta_sm[:, i] = smooth_scores(norm_test_delta[:, i], before_num=b_num)
        norm_val_delta_sm[:, i] = smooth_scores(norm_val_delta[:, i], before_num=b_num)

    return norm_test_delta_sm, norm_val_delta_sm


def smooth_scores(target_data, before_num):

    assert len(target_data.shape) == 1
    sm_scores_ = np.convolve(target_data, np.ones(before_num + 1) / (before_num + 1), mode='valid')
    sm_scores = np.concatenate((target_data[:before_num], sm_scores_), axis=0)

    return sm_scores


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[1]

    topk_indices = np.argpartition(total_err_scores,
                                   range(total_features - topk - 1, total_features), axis=1)[:, -topk:]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=1), axis=1)

    threshold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    metrics = calc_point2point(np.array(gt_labels), pred_labels)

    return metrics, threshold, topk_indices


def get_best_performance_data(total_err_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[1]

    topk_indices = np.argpartition(total_err_scores,
                                   range(total_features - topk - 1, total_features), axis=1)[:, -topk:]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=1), axis=1)

    final_topk_fmeas, thresholds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    threshold = thresholds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    metrics = calc_point2point(np.array(gt_labels), pred_labels)

    return metrics, threshold, topk_indices


def calc_point2point(actual, predict):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    return f1, precision, recall, TP, TN, FP, FN


"""
def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores

def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-1 #default 1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 7 #default 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    return smoothed_err_scores


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, 
        index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas
"""
