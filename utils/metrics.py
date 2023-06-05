# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, mean_squared_error, mean_absolute_error
from utils.create_splits import LABEL_DICT
from utils.pl_utils import to_numpy

def CCC_unbiased(y_pred, y_true, output_single=True, dim=0, keepdim=True):
    '''
        Compute CCC: https://github.com/audeering/audtorch/blob/0.1.1/audtorch/metrics/functional.py#L83
        Args:
            y_true: [#B, #n_emo]
            y_pred: [#B, #n_emo]
    '''
    if y_pred.dim() > 2:  # gather: (bsz, n_labels) -> (world_size, bsz, n_labels)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])

    def _get_moments(data_tensor, dim=dim, keepdim=keepdim):
        mean_t = torch.nanmean(data_tensor, dim=dim, keepdim=keepdim)
        # var_t = torch.var(data_tensor, dim=dim, keepdim=keepdim)
        var_t = 1.0 / (len(data_tensor) - 1) * torch.nansum((data_tensor - mean_t) ** 2)
        return mean_t, var_t
    if y_true.shape[0] == 1:
        # pl_print(f"Batch size must bigger than 1 to calculate var, but get {y_true.shape}")
        ccc = torch.abs(y_pred - y_true).nanmean(dim=dim, keepdim=keepdim)
        ccc_mean = ccc.mean()
        if output_single:
            return ccc_mean, ccc
        return ccc_mean
    gt_mean, gt_var = _get_moments(y_true)
    pred_mean, pred_var = _get_moments(y_pred)

    cov = torch.nansum((y_pred - pred_mean) * (y_true - gt_mean), dim=dim, keepdim=keepdim) / (y_true.shape[0] - 1)
    ccc = (2 * cov) / (pred_var + gt_var + (pred_mean - gt_mean) ** 2).clamp_(1e-6)
    ccc_mean = ccc.mean()
    if output_single:
        return ccc_mean, ccc
    return ccc_mean

def CCC(pred, gt, output_single=False):
    '''
        Compute CCC from https://github.com/Aria-K-Alethia/ExVo/blob/main/metric.py
        Args:
            pred: [#B, #n_emo]
            gt: [#B, #n_emo]
    '''
    pred_mean = pred.mean(0)
    gt_mean = gt.mean(0)

    pred_var = pred.var(0, unbiased=False)
    gt_var = gt.var(0, unbiased=False)

    covar = (pred * gt).mean(0) - pred_mean * gt_mean
    ccc = 2 * covar / (pred_var + gt_var + (pred_mean - gt_mean)**2)
    ccc_mean = ccc.mean()

    #covar2 = torch.mean((pred - pred_mean.unsqueeze(0)) * (gt - gt_mean.unsqueeze(0)), 0)
    #ccc2 = 2 * covar2 / (pred_var + gt_var + (pred_mean - gt_mean)**2)
    if output_single:
        return ccc_mean, ccc
    return ccc_mean

def CCC1(y_true, y_pred):
    """ Official biased CCC function. """
    x_mean = np.nanmean(y_true, dtype="float32")
    y_mean = np.nanmean(y_pred, dtype="float32")
    x_var = 1.0 / (len(y_true) - 1) * np.nansum((y_true - x_mean) ** 2)
    y_var = 1.0 / (len(y_pred) - 1) * np.nansum((y_pred - y_mean) ** 2)
    cov = np.nanmean((y_true - x_mean) * (y_pred - y_mean))
    return round((2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)

class EvalMetrics:
    def CCC(y_true, y_pred, single=False):
        if y_true.reshape(-1).shape == len(y_true):
            return CCC1(y_true, y_pred)
        results = []
        for i in range(y_true.shape[-1]):
            results.append(CCC1(y_true[:, i], y_pred[:, i]))
        if single:
            return np.mean(results), results
        return np.mean(results)

    def MAE(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def MSE(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def UAR(y_true, y_pred):
        return recall_score(y_true, y_pred, average="macro")


class StorePredictions:
    def storehigh(
        task,
        test_pred,
        test_file_ids,
        log_dir,
        suffix="",
    ):
        test_dict_info = {"File_ID": list(test_file_ids)} | {k: to_numpy(test_pred[:, i]).astype(np.float32) for i, k in enumerate(LABEL_DICT["high"][1])}
        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            os.path.join(log_dir, f"Test_A-VB_{task}{suffix}.csv"),
            index=False,
        )

    def storetwo(
        task,
        test_pred,
        test_file_ids,
        log_dir,
        suffix="",
    ):
        test_dict_info = {"File_ID": list(test_file_ids)} | {k: to_numpy(test_pred[:, i]).astype(np.float32) for i, k in enumerate(LABEL_DICT["two"][1])}
        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            os.path.join(log_dir, f"Test_A-VB_{task}{suffix}.csv"),
            index=False,
        )

    def storeculture(
        task,
        test_pred,
        test_file_ids,
        log_dir,
        suffix="",
    ):
        test_dict_info = {"File_ID": list(test_file_ids)} | {k: to_numpy(test_pred[:, i]).astype(np.float32) for i, k in enumerate(LABEL_DICT["culture"][1])}
        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            os.path.join(log_dir, f"Test_A-VB_{task}{suffix}.csv"),
            index=False,
        )

    def storetype(
        task,
        test_pred,
        test_file_ids,
        log_dir,
        le=None,
        suffix="",
    ):
        t_pred = torch.max(test_pred, 1)
        t_pred = le.inverse_transform(t_pred.indices.cpu())
        test_dict_info = {
            "File_ID": list(test_file_ids),
            "Voc_Type": t_pred,
        }
        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            os.path.join(log_dir, "Test_A-VB_{task}_{suffix}.csv)"),
            index=False,
        )


def store_results(task, test_pred, test_file_ids, le=None, log_dir=".", suffix=""):
    os.makedirs(log_dir, exist_ok=True)
    print(task, test_pred.shape, len(test_file_ids))
    if task == "high":
        StorePredictions.storehigh(
            task,
            test_pred,
            test_file_ids,
            log_dir=log_dir,
            suffix=suffix,
        )
    if task == "two":
        StorePredictions.storetwo(
            task,
            test_pred,
            test_file_ids,
            log_dir=log_dir,
            suffix=suffix,
        )
    if task == "culture":
        StorePredictions.storeculture(
            task,
            test_pred,
            test_file_ids,
            log_dir=log_dir,
            suffix=suffix,
        )
    if task == "type":
        StorePredictions.storetype(
            task,
            test_pred,
            test_file_ids,
            le=le,
            log_dir=log_dir,
            suffix=suffix,
        )


if __name__ == "__main__":
    x = torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.4, 0.6, 0.8, 0.9]])
    y = torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.2, 0.6, 0.8, 0.9]])
    # x = torch.rand(16, 10)
    # y = x
    print(CCC(x, y, output_single=True))
    print(EvalMetrics.CCC(x, y))
