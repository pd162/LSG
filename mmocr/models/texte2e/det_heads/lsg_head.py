import torch
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential
from mmdet.models.builder import HEADS, build_loss
import cv2
import numpy as np
from mmocr.models.textdet.dense_heads.head_mixin import HeadMixin
from mmdet.core import multi_apply


@HEADS.register_module()
class LSGDetHead(BaseModule):
    """The class for DBNet head.

    This was partially adapted from https://github.com/MhLiao/DB
    """

    def __init__(self,
                 in_channels,
                 with_bias=False,
                 # decoding_type='db',
                 # text_repr_type='poly',
                 downsample_ratio=4.0,
                 alpha=1.0,
                 beta=1.0,
                 score_thr=0.7,
                 loss=dict(type='LSGDetLoss'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv'),
                     dict(
                         type='Constant', layer='BatchNorm', val=1., bias=1e-4)
                 ]):
        """Initialization.

        Args:
            in_channels (int): The number of input channels of the db head.
            decoding_type (str): The type of decoder for dbnet.
            text_repr_type (str): Boundary encoding type 'poly' or 'quad'.
            downsample_ratio (float): The downsample ratio of ground truths.
            loss (dict): The type of loss for dbnet.
        """
        # super().__init__(init_cfg=init_cfg)
        super().__init__()

        assert isinstance(in_channels, int)

        self.in_channels = in_channels
        self.out_channel = 3
        # self.text_repr_type = text_repr_type
        self.loss_module = build_loss(loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        # self.decoding_type = decoding_type

        self.convs = Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.Conv2d(
                in_channels, in_channels, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid()
            nn.Conv2d(
                in_channels, self.out_channel, 3, bias=with_bias, padding=1),
        )


    def forward(self, inputs):
        if not isinstance(inputs, tuple):
            inputs = [inputs]
        outputs = multi_apply(self.forward_single, inputs)
        return outputs

    def forward_single(self, inputs):
        outputs = self.convs(inputs)
        return outputs,

    def get_start_points(self, pred_maps):
        # pass
        pred_maps = pred_maps[0]
        assert pred_maps.shape[0] == 1
        text_region_map = torch.sigmoid(pred_maps[:, 0, :, :])
        text_head_map = torch.sigmoid(pred_maps[:, 1, :, :])
        text_center_map = torch.sigmoid(pred_maps[:, 2, :, :])
        text_region_map = (text_region_map * text_head_map)
        head_score_map = ((text_center_map ** self.beta) * (text_region_map ** self.beta)) ** (
                    1 / (self.alpha + self.beta))  # 这里为何都是beta

        # text_region_map = ((text_head_map * text_region_map) + text_region_map) / 2
        # text_center_map = text_center_map * text_head_map

        # head_score_map = (text_region_map * text_head_map * text_center_map) ** (1/2)

        # text_region_map = text_head_map * text_region_map
        # head_score_map = ((text_center_map ** self.beta) * (text_region_map ** self.beta)) ** (
        #             1 / (self.alpha + self.beta))  # 这里为何都是beta
        # head_score_map = (text_region_map + head_score_map) / 2
        # head_score_map = (text_region_map * text_center_map) ** (1./2)
        # head_score_map = text_region_map
        # head_score_map = (text_center_map * self.alpha + text_region_map * self.beta) / (self.alpha + self.beta)
        # head_score_map = text_center_map * text_region_map
        # head_score_map = text_center_map * text_region_map * 2 / (text_center_map + text_region_map)  # 几种平均数都差不多
        head_score_map = head_score_map.cpu().numpy()[0]
        head_pred_mask = head_score_map >= self.score_thr
        head_mask = fill_hole(head_pred_mask)

        head_contours, _ = cv2.findContours(
            head_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(head_mask)
        reference_points = []
        reference_points_score = []
        for head in head_contours:
            cnt_head_mask = mask.copy().astype(np.int8)
            cv2.drawContours(cnt_head_mask, [head], -1, 1, -1)
            if (cnt_head_mask.sum()) < 1:
                continue
            cnt_head_score = text_head_map.cpu().numpy()[0] * cnt_head_mask
            # max_xy = np.argmax(cnt_head_score)
            cnt_head_xy = np.argwhere(cnt_head_score >= cnt_head_score.max() * 0.98)
            #
            det_score = head_score_map[cnt_head_score >= cnt_head_score.max() * 0.98].mean()
            head_point = cnt_head_xy.mean(axis=0)
            head_point = head_point[::-1]
            reference_points.append(head_point)
            reference_points_score.append(cnt_head_score.max() * det_score)

        reference_points = torch.tensor(np.array(reference_points), dtype=pred_maps.dtype, device=pred_maps.device)
        reference_points = reference_points * self.downsample_ratio
        reference_points_score = torch.tensor(reference_points_score, dtype=pred_maps.dtype, device=pred_maps.device)
        results = {
            "reference_points": reference_points.view(-1, 1, 2).cpu().numpy(),
            "reference_points_score": reference_points_score
        }

        return results

    # def loss(self, pred_maps, **kwargs):
    def loss(self, pred_maps, **kwargs):
        """Compute the loss for scene text detection.

        Args:
            pred_maps (Tensor): The input score maps of shape
                :math:`(NxCxHxW)`.

        Returns:
            dict: The dict for losses.
        """
        losses = self.loss_module(pred_maps, self.downsample_ratio, **kwargs)

        return losses


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool_)

    return ~canvas | input_mask
