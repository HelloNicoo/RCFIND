# -*- coding: utf-8 -*-
# @Time    : 2022/8/3 14:08
# @Author  : sylviazz
# @FileName: metric
# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:39
# @Author  : sylviazz
# @FileName: metrics

class DRScore(object):
    """
    aspect, opinion,(aspect, opinion) pair, (aspect, sentiment) pair, (aspect, opinion, sentiment) triplet,
    (aspect, category, opinion, sentiemnt) quadruple指标计算
    """

    def __init__(self):
        self.true_rel = .0
        self.pred_rel = .0
        self.gold_rel = .0
    def compute(self):
        # aspect
        asp_p = 0 if self.true_asp + self.pred_asp == 0 else 1. * self.true_asp / self.pred_asp
        asp_r = 0 if self.true_asp + self.gold_asp == 0 else 1. * self.true_asp / self.gold_asp
        asp_f = 0 if asp_p + asp_r == 0 else 2 * asp_p * asp_r / (asp_p + asp_r)
        print({"aspect": {"true_asp": self.true_asp, "pred_asp": self.pred_asp, "gold_asp": self.gold_asp}})
        return {"aspect": {"precision": asp_p, "recall": asp_r, "f1": asp_f}}
    def update_rel(self, pred_rel, gold_rel):
        if pred_rel != 0:
            self.pred_rel += 1
        if gold_rel != 0:
            self.gold_rel += 1
        if (pred_rel != 0) and (gold_rel != 0) and (pred_rel == gold_rel):
            self.true_rel += 1
        pre = 0 if self.true_rel + self.pred_rel == 0 else 1. * self.true_rel / self.pred_rel
        rec = 0 if self.true_rel + self.gold_rel == 0 else 1. * self.true_rel / self.gold_rel
        f1 = 0 if pre + rec == 0 else 2 * pre * rec / (pre + rec)
        return pre, rec, f1