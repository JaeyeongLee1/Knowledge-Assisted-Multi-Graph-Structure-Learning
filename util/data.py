from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile


def eval_scores(scores, true_labels, th_steps, return_thresold=False):
    padding_list = [0]*(len(true_labels) - len(scores))

    if len(padding_list) > 0:
        scores = padding_list + scores
        print('add padding list to score list')

    scores_sorted = rankdata(scores, method='ordinal')
    th_vals = np.array(range(1, th_steps)) * 1.0 / th_steps

    fmeas = [None] * (th_steps - 1)
    thresholds = [None] * (th_steps - 1)

    for i in range(th_steps - 1):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_labels, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds

    return fmeas


"""
def get_attack_interval(attack):
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i - 1] == 0:
                heads.append(i)

            if i < len(attack) - 1 and attack[i + 1] == 0:
                tails.append(i)
            elif i == len(attack) - 1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res


def eval_mseloss(predicted, ground_truth):

    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss


def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_err_median_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta


def get_err_mean_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta


def get_err_mean_and_std(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):

    padding_list = [0]*(len(gt) - len(scores))

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()

    return f1_score(gt, pred_labels)
"""
