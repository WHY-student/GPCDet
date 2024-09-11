# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .assign_result_neu import AssignResultNeu
from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

INF = 100000000


@BBOX_ASSIGNERS.register_module()
class TaskNeuAssigner2(BaseAssigner):
    """Task aligned assigner used in the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.

    Assign a corresponding gt bbox or background to each predicted bbox.
    Each bbox will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (int): number of bbox selected in each level
        iou_calculator (dict): Config dict for iou calculator.
            Default: dict(type='BboxOverlaps2D')
    """

    def __init__(self, topk, div=4.0,iou_calculator=dict(type='BboxOverlaps2D'),assign_metric='iou'):
        assert topk >= 1
        self.topk = topk
        self.div = div
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.iou_calculator2 = build_iou_calculator(iou_calculator)
        self.assign_metric2 = assign_metric
        self.assign_metric = assign_metric
    def assign(self,
               pred_scores,  # 预测分数tensor(n,16) n为所有层所有anchor预测分数
               decode_bboxes, # 回归框tensor(n,4)
               anchors,  #anchor (n,4)
               gt_bboxes, # gt(m,4)
               gt_bboxes_ignore=None,
               gt_labels=None, # gtlabel(m)
               alpha=1,
               beta=6):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)


        Args:
            pred_scores (Tensor): predicted class probability,
                shape(n, num_classes)
            decode_bboxes (Tensor): predicted bounding boxes, shape(n, 4)
            anchors (Tensor): pre-defined anchors, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        anchors = anchors[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), anchors.size(0) # numgt (m) numbbox （n）
        # compute alignment metric between all bbox and gt
        overlaps = self.iou_calculator(decode_bboxes, gt_bboxes,mode=self.assign_metric).detach() # overlap tensor  (n,m)
        bbox_scores = pred_scores[:, gt_labels].detach() # (n,m) 就是每个框预测的分数
        # assign 0 by default
        assigned_gt_inds = anchors.new_full((num_bboxes, ), #tensor (n)  每个样本对应的gt索引+1。 -1表示忽略样本 ; 0表示负样本 ; 1~len(gt)表示正样本。
                                            0, #原始为0
                                            dtype=torch.long)
        assign_metrics = anchors.new_zeros((num_bboxes, )) # tensor(n)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = anchors.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No gt boxes, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = anchors.new_full((num_bboxes, ),  # ，表示每个样本对应的label类别。-1表示非正样本(包括负样本和忽略样本)，0~len(class)-1就是正样本对应的label。
                                                   -1,
                                                   dtype=torch.long)
            assign_result = AssignResultNeu(
                num_gt, assigned_gt_inds, max_overlaps,max_overlaps,labels=assigned_labels)
            assign_result.assign_metrics = assign_metrics
            return assign_result

        # select top-k bboxes as candidates for each gt
        alignment_metrics = bbox_scores**alpha * overlaps**beta  # tensor(n,m)
        topk = min(self.topk, alignment_metrics.size(0))
        _, candidate_idxs = alignment_metrics.topk(topk, dim=0, largest=True)  #tensor(topk,m) m是gt数量
        candidate_metrics = alignment_metrics[candidate_idxs,
                                              torch.arange(num_gt)]
        is_pos = candidate_metrics > 0




        #  # limit the neu sample's center in bbox
        anchors_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        anchors_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0

        # 计算新anchor的宽度和高度
        widths = (anchors[:, 2] - anchors[:, 0]) / self.div
        heights = (anchors[:, 3] - anchors[:, 1]) / self.div

        # 构建新的anchor
        new_anchors = torch.zeros_like(anchors)
        new_anchors[:, 0] = anchors_cx - widths / 2.0
        new_anchors[:, 1] = anchors_cy - heights / 2.0
        new_anchors[:, 2] = anchors_cx + widths / 2.0
        new_anchors[:, 3] = anchors_cy + heights / 2.0

        overlaps2 = self.iou_calculator(new_anchors, gt_bboxes, mode=self.assign_metric).detach()
        overlaps2 = overlaps2[candidate_idxs, torch.arange(num_gt)]
        is_in_bbox = overlaps2 > 0

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_anchors_cx = anchors_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_anchors_cy = anchors_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_anchors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_anchors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_anchors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_anchors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

        is_pos = is_pos & is_in_gts
        is_mid=is_in_bbox & ~is_pos


        # 我写的
        # indexmid = t[~is_pos.view(-1)]
        # assigned_gt_inds[indexmid]=-1

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        # 赋予初值 tensor(n*m) 都为_INF
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]

        overlaps_inf2=torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        # 忽略样本也分~（gt+1）
        indexmid = candidate_idxs.view(-1)[is_mid.view(-1)]
        overlaps_inf2[indexmid] = overlaps.t().contiguous().view(-1)[indexmid]

        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        overlaps_inf2 = overlaps_inf2.view(num_gt, -1).t()

        # 求每个gt最大的iou的值以及下标
        max_overlaps2, argmax_overlaps2 = overlaps_inf2.max(dim=1)  # 压缩成一列,求出每个anchor与那个iou最大的gt，对于每个anchor都有一个对应最大iou gt tensor(n)
        assigned_gt_inds[max_overlaps2 != -INF] = -argmax_overlaps2[
                                                       max_overlaps2 != -INF] - 1  # 最大iou不为inf 将这个anchor与其最大的IOU的gt赋予下标-(gt+1) 正样本
        assign_metrics[max_overlaps2 != -INF] = alignment_metrics[
            max_overlaps2 != -INF, argmax_overlaps2[max_overlaps2 != -INF]]

        # 求每个gt最大的iou的值以及下标
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1  # 赋予gt+1 正样本
        assign_metrics[max_overlaps != -INF] = alignment_metrics[
            max_overlaps != -INF, argmax_overlaps[max_overlaps != -INF]]
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
            nue_inds=torch.nonzero(
                assigned_gt_inds < 0, as_tuple=False).squeeze()
            if nue_inds.numel() > 0:
                assigned_labels[nue_inds] = gt_labels[
                    -assigned_gt_inds[nue_inds] - 1]
        else:
            assigned_labels = None
        assign_result = AssignResultNeu(
            num_gt, assigned_gt_inds, max_overlaps,max_overlaps2, labels=assigned_labels)
        assign_result.assign_metrics = assign_metrics
        return assign_result
