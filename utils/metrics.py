# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:39
# @Author  : sylviazz
# @FileName: metrics

class MRCScore(object):
    """
    aspect, opinion,(aspect, opinion) pair, (aspect, sentiment) pair, (aspect, opinion, sentiment) triplet,
    (aspect, category, opinion, sentiemnt) quadruple指标计算
    """

    def __init__(self):
        # aspect
        self.true_asp = .0
        self.pred_asp = .0
        self.gold_asp = .0
        # opinion
        self.true_opi = .0
        self.pred_opi = .0
        self.gold_opi = .0
        # (aspect, opinion) pair
        self.true_ao_pair = .0
        self.pred_ao_pair = .0
        self.gold_ao_pair = .0
        # (aspect, sentiment) pair
        self.true_as_pair = .0
        self.pred_as_pair = .0
        self.gold_as_pair = .0
        # (aspect, opinion, sentiment) triplet
        self.true_triplet = .0
        self.pred_triplet = .0
        self.gold_triplet = .0
        # (aspect, category, opinion, sentiemnt) quadruple
        self.true_quadruple = .0
        self.pred_quadruple = .0
        self.gold_quadruple = .0
        # rel
        self.true_rel = .0
        self.pred_rel = .0
        self.gold_rel = .0
    def compute(self):
        # aspect
        asp_p = 0 if self.true_asp + self.pred_asp == 0 else 1. * self.true_asp / self.pred_asp
        asp_r = 0 if self.true_asp + self.gold_asp == 0 else 1. * self.true_asp / self.gold_asp
        asp_f = 0 if asp_p + asp_r == 0 else 2 * asp_p * asp_r / (asp_p + asp_r)
        print({"aspect": {"true_asp": self.true_asp, "pred_asp": self.pred_asp, "gold_asp": self.gold_asp}})
        # opinion
        opi_p = 0 if self.true_opi + self.pred_opi == 0 else 1. * self.true_opi / self.pred_opi
        opi_r = 0 if self.true_opi + self.gold_opi == 0 else 1. * self.true_opi / self.gold_opi
        opi_f = 0 if opi_p + opi_r == 0 else 2 * opi_p * opi_r / (opi_p + opi_r)
        print({"opinion": {"true_opi": self.true_opi, "pred_opi": self.pred_opi, "gold_opi": self.gold_opi}})
        # (aspect, opinion) pair
        ao_pair_p = 0 if self.true_ao_pair + self.pred_ao_pair == 0 else 1. * self.true_ao_pair / self.pred_ao_pair
        ao_pair_r = 0 if self.true_ao_pair + self.gold_ao_pair == 0 else 1. * self.true_ao_pair / self.gold_ao_pair
        ao_pair_f = 0 if ao_pair_p + ao_pair_r == 0 else 2 * ao_pair_p * ao_pair_r / (ao_pair_p + ao_pair_r)
        print({"ao_pair": {"true_ao_pair": self.true_ao_pair, "pred_ao_pair": self.pred_ao_pair, "gold_ao_pair": self.gold_ao_pair}})
        # (aspect, sentiment) pair
        as_pair_p = 0 if self.true_as_pair + self.pred_as_pair == 0 else 1. * self.true_as_pair / self.pred_as_pair
        as_pair_r = 0 if self.true_as_pair + self.gold_as_pair == 0 else 1. * self.true_as_pair / self.gold_as_pair
        as_pair_f = 0 if as_pair_p + as_pair_r == 0 else 2 * as_pair_p * as_pair_r / (as_pair_p + as_pair_r)
        print({"as_pair": {"true_as_pair": self.true_as_pair, "pred_as_pair": self.pred_as_pair, "gold_as_pair": self.gold_as_pair}})
        # (aspect, opinion, sentiment) triplet
        triplet_p = 0 if self.true_triplet + self.pred_triplet == 0 else 1. * self.true_triplet / self.pred_triplet
        triplet_r = 0 if self.true_triplet + self.gold_triplet == 0 else 1. * self.true_triplet / self.gold_triplet
        triplet_f = 0 if triplet_p + triplet_r == 0 else 2 * triplet_p * triplet_r / (triplet_p + triplet_r)
        print({"triplet": {"true_triplet": self.true_triplet, "pred_triplet": self.pred_triplet, "gold_triplet": self.gold_triplet}})
        # (aspect, category, opinion, sentiemnt) quadruple
        quadruple_p = 0 if self.true_quadruple + self.pred_quadruple == 0 else 1. * self.true_quadruple / self.pred_quadruple
        quadruple_r = 0 if self.true_quadruple + self.gold_quadruple == 0 else 1. * self.true_quadruple / self.gold_quadruple
        quadruple_f = 0 if quadruple_p + quadruple_r == 0 else 2 * quadruple_p * quadruple_r / (quadruple_p + quadruple_r)
        print({"quadruple": {"true_quadruple": self.true_quadruple, "pred_quadruple": self.pred_quadruple, "gold_quadruple": self.gold_quadruple}})

        return {"aspect": {"precision": asp_p, "recall": asp_r, "f1": asp_f},
                "opinion": {"precision": opi_p, "recall": opi_r, "f1": opi_f},
                "ao_pair": {"precision": ao_pair_p, "recall": ao_pair_r, "f1": ao_pair_f},
                "as_pair": {"precision": as_pair_p, "recall": as_pair_r, "f1": as_pair_f},
                "triplet": {"precision": triplet_p, "recall": triplet_r, "f1": triplet_f},
                "quadruple": {"precision": quadruple_p, "recall": quadruple_r, "f1": quadruple_f}}
    def update_rel(self, pred_rel, gold_rel):
         self.gold_rel += len(gold_rel)
         self.pred_rel += len(pred_rel)
         for g in gold_rel:
            for p in pred_rel:
                if g == p:
                    self.true_rel += 1
         pre = 0 if self.true_rel + self.pred_rel == 0 else 1. * self.true_rel / self.pred_rel
         rec = 0 if self.true_rel + self.gold_rel == 0 else 1. * self.true_rel / self.gold_rel
         f1 = 0 if pre + rec == 0 else 2 * pre * rec / (pre + rec)
         return pre, rec, f1
    def update(self, gold_aspects, gold_opinions, gold_ao_pairs, gold_as_pairs, gold_triplets, gold_quadruples,
               pred_aspects, pred_opinions, pred_ao_pairs, pred_as_pairs, pred_triplets, pred_quadruples, asp_pree, opi_pree):
        self.gold_asp += len(gold_aspects)
        self.gold_opi += len(gold_opinions)
        self.gold_ao_pair += len(gold_ao_pairs)
        self.gold_as_pair += len(gold_as_pairs)
        self.gold_triplet += len(gold_triplets)
        self.gold_quadruple += len(gold_quadruples)
        self.pred_asp += len(asp_pree)
        self.pred_opi += len(opi_pree)
        self.pred_ao_pair += len(pred_ao_pairs)
        self.pred_as_pair += len(pred_as_pairs)
        self.pred_triplet += len(pred_triplets)
        self.pred_quadruple += len(pred_quadruples)
        for g in gold_aspects:
            for p in asp_pree:
                if g == p:
                    self.true_asp += 1

        for g in gold_opinions:
            for p in opi_pree:
                if g == p:
                    self.true_opi += 1

        self.gold_rel += len(gold_rel)
        self.pred_rel += len(pred_rel)
        for g in gold_rel:
           for p in pred_rel:
               if g == p:
                   self.true_rel += 1
        pre = 0 if self.true_rel + self.pred_rel == 0 else 1. * self.true_rel / self.pred_rel
        rec = 0 if self.true_rel + self.gold_rel == 0 else 1. * self.true_rel / self.gold_rel
        f1 = 0 if quadruple_p + quadruple_r == 0 else 2 * quadruple_p * quadruple_r / (quadruple_p + quadruple_r)
        return pre, rec, f1
