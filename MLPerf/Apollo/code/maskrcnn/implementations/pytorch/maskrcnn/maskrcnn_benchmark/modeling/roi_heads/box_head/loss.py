# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_iou_batched
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from torch.nn.utils.rnn import pad_sequence

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.const0123, self.const4567 = None, None
        self.batch_arange = None
        self.syncfree = True

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def match_targets_to_proposals_batched(self, proposal, target):
        match_quality_matrix = boxlist_iou_batched(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix, batched=1)
        # Fast RCNN only need "labels" field for selecting the targets
        # how to do this for batched case?
        # target = target.copy_with_fields("labels")
        return matched_idxs

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        matched_idxs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets_per_image = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs_per_image = matched_targets_per_image.get_field("matched_idxs")

            labels_per_image = matched_targets_per_image.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_per_image == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image.masked_fill_(bg_inds, 0)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
            labels_per_image.masked_fill(ignore_inds, -1)  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets_per_image.bbox, proposals_per_image.bbox
            )
   
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_idxs.append(matched_idxs_per_image)
        return labels, regression_targets, matched_idxs

    def prepare_targets_batched(self, proposals, targets, target_labels):
        num_images = proposals.size(0)
        matched_idxs = self.match_targets_to_proposals_batched(proposals, targets)
        img_idx = torch.arange(num_images, device = proposals.device)[:, None]
        labels = target_labels[img_idx, matched_idxs.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels.masked_fill_(bg_inds, 0)
        ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels.masked_fill_(ignore_inds, -1)

        matched_targets = targets[img_idx, matched_idxs.clamp(min=0)]

        regression_targets = self.box_coder.encode(
            matched_targets.view(-1,4), proposals.view(-1,4)
        )
        return labels, regression_targets.view(num_images, -1, 4), matched_idxs

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        num_images = len(proposals[0])
        target_boxes, _, target_labels, _ = targets
        prop_boxes, prop_scores, image_sizes = proposals[0], proposals[1], proposals[2]
        labels, regression_targets, matched_idxs = self.prepare_targets_batched(prop_boxes, target_boxes, target_labels)
          
        # scores is used as a mask, -1 means box is invalid
        if self.syncfree and num_images == 1:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, is_rpn=0, objectness=prop_scores)
            device = sampled_pos_inds[0].device
            import maskrcnn_benchmark.Syncfree
            inds, counts = maskrcnn_benchmark.Syncfree.balanced_pos_neg_sampler_repeat(
                    sampled_pos_inds[0], torch.empty([0], device=device, dtype=torch.int64),
                    sampled_neg_inds[0], torch.empty([0], device=device, dtype=torch.int64),
                    self.fg_bg_sampler.batch_size_per_image,
                    self.fg_bg_sampler.batch_size_per_image,
                    True)
            if self.batch_arange is None:
                self.batch_arange = torch.arange(0,self.fg_bg_sampler.batch_size_per_image,1, device=device)
            sampled_pos_inds_mask = (self.batch_arange < counts[0]).unsqueeze(1)
            prop_boxes = prop_boxes.view(-1,4)
            regression_targets = regression_targets.view(-1,4)
            labels = labels.view(-1)
            matched_idxs = matched_idxs.view(-1)
            result_proposals = []
            for i in range(num_images):
                box = BoxList(prop_boxes[inds], image_size = image_sizes[i])
                box.add_field("matched_idxs", matched_idxs[inds])
                box.add_field("regression_targets", regression_targets[inds])
                box.add_field("labels", labels[inds])
                result_proposals.append(box)
            self._proposals = result_proposals
            return result_proposals
        else:
            if num_images == 1:
                sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, is_rpn=0, objectness=prop_scores)
                # when num_images=1, sampled pos inds only has 1 item, so avoid copy in torch.cat
                with torch.cuda.nvtx.range("NZ2"):
                    pos_inds_per_image = [torch.nonzero(sampled_pos_inds[0]).squeeze(1)]
                    neg_inds_per_image = [torch.nonzero(sampled_neg_inds[0]).squeeze(1)]
            else:
                sampled_pos_inds, sampled_neg_inds, num_pos_samples, num_neg_samples = self.fg_bg_sampler(labels, is_rpn=0, objectness=prop_scores)
                pos_inds_per_image = sampled_pos_inds.split(list(num_pos_samples))
                neg_inds_per_image = sampled_neg_inds.split(list(num_neg_samples))
            prop_boxes = prop_boxes.view(-1,4)
            regression_targets = regression_targets.view(-1,4)
            labels = labels.view(-1)
            matched_idxs = matched_idxs.view(-1)
            result_proposals = []
            for i in range(num_images):
                inds = torch.cat([pos_inds_per_image[i], neg_inds_per_image[i]])
                box = BoxList(prop_boxes[inds], image_size = image_sizes[i])
                box.add_field("matched_idxs", matched_idxs[inds])
                box.add_field("regression_targets", regression_targets[inds])
                box.add_field("labels", labels[inds])
                result_proposals.append(box)
            self._proposals = result_proposals
            return result_proposals

    def compute_box_loss(self, labels, box_regression, regression_targets):
        with torch.cuda.nvtx.range("NZ4"):
            sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels.index_select(0, sampled_pos_inds_subset)
        if self.cls_agnostic_bbox_reg:
            if self.const4567 is None:
                self.const4567 = torch.tensor([4, 5, 6, 7], pin_memory=True).cuda(non_blocking=True)
            map_inds = self.const4567
        else:
            with torch.cuda.nvtx.range("H2D1"):
                if self.const0123 is None:
                    self.const0123 = torch.tensor([0, 1, 2, 3], pin_memory=True).cuda(non_blocking=True)
                map_inds = 4 * labels_pos[:, None] + self.const0123

        index_select_indices=((sampled_pos_inds_subset[:,None]) * box_regression.size(1) + map_inds).view(-1)
        box_regression_sampled=box_regression.view(-1).index_select(0, index_select_indices).view(map_inds.shape[0], 
                                                                                                  map_inds.shape[1]) 
        regression_targets_sampled = regression_targets.index_select(0, sampled_pos_inds_subset)
        #print("\nlabels = %s\nsampled_pos_inds_subset = %s\nlabels_pos = %s\nmap_inds = %s\nindex_select_indices = %s\nbox_regression_sampled = %s\nregression_targets_sampled = %s\n" % (str(labels), str(sampled_pos_inds_subset), str(labels_pos), str(map_inds), str(index_select_indices), str(box_regression_sampled), str(regression_targets_sampled)))

        box_loss = smooth_l1_loss(
            box_regression_sampled,
            regression_targets_sampled,
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()
        return box_loss

    def compute_box_loss_sf(self, labels, box_regression, regression_targets):
        import maskrcnn_benchmark.Syncfree
        sampled_pos_inds_subset, counts = maskrcnn_benchmark.Syncfree.nonzero_repeat(labels > 0, torch.tensor([], device=labels.device, dtype=torch.int64))
        sampled_pos_inds_mask = torch.arange(0,labels.numel(),1, device=labels.device, dtype=torch.int32)
        sampled_pos_inds_mask = sampled_pos_inds_mask < counts
        sampled_pos_inds_mask = sampled_pos_inds_mask.unsqueeze(1)

        labels_pos = labels.index_select(0, sampled_pos_inds_subset)
        if self.cls_agnostic_bbox_reg:
            if self.const4567 is None:
                self.const4567 = torch.tensor([4, 5, 6, 7], pin_memory=True).cuda(non_blocking=True)
            map_inds = self.const4567
        else:
            with torch.cuda.nvtx.range("H2D1"):
                if self.const0123 is None:
                    self.const0123 = torch.tensor([0, 1, 2, 3], pin_memory=True).cuda(non_blocking=True)
                map_inds = 4 * labels_pos[:, None] + self.const0123

        index_select_indices=((sampled_pos_inds_subset[:,None]) * box_regression.size(1) + map_inds).view(-1)
        box_regression_sampled=box_regression.view(-1).index_select(0, index_select_indices).view(map_inds.shape[0], 
                                                                                                  map_inds.shape[1]) 
        regression_targets_sampled = regression_targets.index_select(0, sampled_pos_inds_subset)

        box_loss = smooth_l1_loss(
            box_regression_sampled * sampled_pos_inds_mask,
            regression_targets_sampled * sampled_pos_inds_mask,
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()
        return box_loss

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        if self.syncfree:
            box_loss = self.compute_box_loss_sf(labels, box_regression, regression_targets)
        else:
            box_loss = self.compute_box_loss(labels, box_regression, regression_targets)

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
