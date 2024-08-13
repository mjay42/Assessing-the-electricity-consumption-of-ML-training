# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
#from apex.contrib.conv_bias_relu import ConvBiasReLU, ConvBias
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor
from maskrcnn_benchmark.layers.nhwc import Conv2d_NHWC, nhwc_to_nchw_transform, nchw_to_nhwc_transform
from maskrcnn_benchmark.layers.nhwc import init
from maskrcnn_benchmark.utils.mlperf_logger import mllogger
from maskrcnn_benchmark.utils.fuse_conv import ConvBiasReLU_



ConvBiasReLU = ConvBiasReLU_.apply


# Disable for now due to cudnn issue
#ConvBias = ConvBias_.apply


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.nhwc = cfg.NHWC
        self.fusion = cfg.MODEL.RPN.FUSION
        conv = Conv2d_NHWC if self.nhwc else nn.Conv2d
        self.conv = conv(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = conv(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = conv(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        mllogger.event(mllogger.constants.WEIGHTS_INITIALIZATION, metadata=dict(tensor='RPNHead_conv'))
        mllogger.event(mllogger.constants.WEIGHTS_INITIALIZATION, metadata=dict(tensor='RPNHead_cls'))
        mllogger.event(mllogger.constants.WEIGHTS_INITIALIZATION, metadata=dict(tensor='RPNHead_bbox'))

    def forward(self, x):
        logits = []
        bbox_regs = []

        if self.fusion:
            conv_w = self.conv.weight.half()
            conv_b = self.conv.bias.reshape(1, -1, 1, 1).half()

            #bbox_pred_w = self.bbox_pred.weight.half()
            #bbox_pred_b = self.bbox_pred.bias.reshape(1, -1, 1, 1)

            if self.nhwc:
                # Issue: need to convert weight and bias to nchw because ConvBiasReLU and ConvBias, as they need (c_out, c_in, h, w)
                conv_w = torch.permute(conv_w, (0, 3, 1, 2))
                # Issue: fuse doesn't support even number of channels
                # cls_logits_w = torch.permute(cls_logits_w, (0, 3, 1, 2))
                # Issue: will throw cudnn error, disable for now
                # bbox_pred_w = torch.permute(bbox_pred_w, (0, 3, 1, 2)).to(memory_format=torch.channels_last)

        for feature in x:
            if not self.fusion:
                t = F.relu(self.conv(feature))
                logit = self.cls_logits(t)
                bbox_reg = self.bbox_pred(t)
                if self.nhwc:
                    logit = nhwc_to_nchw_transform(logit)
                    bbox_reg = nhwc_to_nchw_transform(bbox_reg)
            else:
                t = ConvBiasReLU(feature, conv_w, conv_b, 1, 1)
                logit = self.cls_logits(t)
                bbox_reg = self.bbox_pred(t)
                # can't use fuse due to cudnn error
                #bbox_reg = ConvBias(t, bbox_pred_w, bbox_pred_b, 0, 1)
                if self.nhwc:
                    logit = nhwc_to_nchw_transform(logit)
                    bbox_reg = nhwc_to_nchw_transform(bbox_reg)

            logits.append(logit)
            bbox_regs.append(bbox_reg)
        return logits, bbox_regs


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.nhwc = cfg.NHWC

    def forward(self, images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # For NHWC case, only RPN head is implemented in NHWC; everything else in RPN is NCHW
        # For NHWC, features is a two element list with features in NHWC and NCHW layout
        batched_anchor_data = [anchor_boxes, anchor_visibility, [tuple(image_size_wh) for image_size_wh in images.image_sizes_wh]]
        if self.training:
            return self._forward_train(batched_anchor_data, objectness, rpn_box_regression, images.image_sizes_tensor, targets)
        else:
            return self._forward_test(batched_anchor_data, objectness, rpn_box_regression, images.image_sizes_tensor)

    def _forward_train(self, anchors, objectness, rpn_box_regression, image_shapes_cat, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, image_shapes_cat, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression, image_shapes_cat):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression, image_shapes_cat)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn_head(cfg):
    """
    Return RPN head only, used when RPN head is included in backbone.
    """
    anchor_generator = make_anchor_generator(cfg)

    in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
    head = rpn_head(
        cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
    )
    return anchor_generator, head


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg)

    return RPNModule(cfg)
