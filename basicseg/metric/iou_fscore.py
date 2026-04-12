import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Binary_metric():
    'calculate fscore and iou'
    def __init__(self, thr=0.3):
        self.mean_reset()
        self.norm_reset()
        self.thr = thr
        self.cnt = 0

    def mean_reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def norm_reset(self):
        self.norm_metric = {'precision':0., 'recall':0., 'fscore':0., 'iou':0.}
        self.cnt = 0.

    def update(self, pred, target):
        # for safety
        pred = pred.detach().clone()
        target = target.detach().clone()

        # 打印完整诊断
        print(f"\n[METRIC] pred: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}")
        print(f"[METRIC] target: shape={target.shape}, min={target.min():.4f}, max={target.max():.4f}")
        print(f"[METRIC] target unique前10个: {torch.unique(target).cpu().numpy()[:10].tolist()}")

        # ====== 1. pred 处理 ======
        if pred.max() > 1.5:
            pred = torch.sigmoid(pred)
        pred = pred.clamp(0, 1)
        print(f"[METRIC] pred after sigmoid: min={pred.min():.4f}, max={pred.max():.4f}")

        # ====== 2. target 处理 ======
        # 简化逻辑：统一用阈值0.5
        target_binary = (target > 0.5).float()
        print(f"[METRIC] target_binary: unique={torch.unique(target_binary).tolist()}, sum={target_binary.sum():.0f}")

        # ====== 3. pred 二值化 ======
        pred_binary = (pred >= self.thr).float()
        print(f"[METRIC] pred_binary: unique={torch.unique(pred_binary).tolist()}, sum={pred_binary.sum():.0f}")

        # ====== 4. 计算 ======
        pred_flat = pred_binary.view(-1).float()
        target_flat = target_binary.view(-1).float()

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()

        print(f"[METRIC] TP={tp:.0f}, FP={fp:.0f}, FN={fn:.0f}, TN={tn:.0f}")

        # ====== 5. 累加 ======
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.cnt += pred.shape[0]

        # ====== 6. norm ======
        eps = 1e-6
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        fscore = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fn + fp + eps)

        self.norm_metric['precision'] += precision
        self.norm_metric['recall'] += recall
        self.norm_metric['fscore'] += fscore
        self.norm_metric['iou'] += iou

        print(f"[METRIC] >>> IoU={iou:.4f}, F1={fscore:.4f}")

    def get_mean_result(self):
        mean_metric = self.mean_compute()
        self.mean_reset()
        return mean_metric

    def get_norm_result(self):
        if self.cnt > 0:
            for k in self.norm_metric.keys():
                self.norm_metric[k] /= self.cnt
        norm_metric = deepcopy(self.norm_metric)
        self.norm_reset()
        return norm_metric

    def mean_compute(self):
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        fscore = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fn + self.fp + eps)
        return {"precision":precision, "recall":recall, "fscore":fscore, "iou":iou}