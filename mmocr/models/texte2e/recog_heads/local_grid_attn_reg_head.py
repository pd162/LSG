import copy
import numpy as np
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32, BaseModule
from torch.nn.init import normal_
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.parallel import DataContainer as DC
from mmdet.models.builder import HEADS
from mmocr.models.builder import DECODERS, build_convertor
from .autoregtf_decoder import AutoRegTF


@HEADS.register_module()
class LocalGridHead(BaseModule):

    def __init__(self,
                 init_cfg=None,
                 embed_dims=256,
                 label_convertor=None,
                 num_feature_levels=4,
                 num_reg_fcs=1,
                 l1_reg=True,
                 with_poly=False,
                 step_as_test=False,
                 reg_loss=dict(type='TFLoss', reduction='mean'),
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 decoder={},

                 ):
        super(LocalGridHead, self).__init__(init_cfg)
        self.convertor = build_convertor(label_convertor)
        self.num_classes = self.convertor.num_classes()
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.num_feature_levels = num_feature_levels
        self.max_length = self.convertor.max_seq_len
        self.positional_encoding = build_positional_encoding(positional_encoding)
        decoder['padding_idx'] = self.convertor.padding_idx
        decoder['start_idx'] = self.convertor.start_idx
        decoder['num_classes'] = self.num_classes
        decoder['max_seq_len'] = self.convertor.max_seq_len
        self.decoder = AutoRegTF(**decoder)
        # self.reg_loss = build_loss(reg_loss)
        self.l1_reg = l1_reg
        self.with_poly = with_poly
        self.step_as_test = step_as_test
        self.shift_from_start = getattr(decoder, 'shift_from_start', False)
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        # fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4 if self.with_poly else 2))
        reg_branch = nn.Sequential(*reg_branch)
        self.reg_branches = reg_branch

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.classifier = nn.Linear(self.embed_dims, self.num_classes)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.decoder.init_weights()
        constant_init(self.reg_branches[-1], 0, bias=0)
        normal_(self.level_embeds)

    def padding_points(self, reference_points_per_img):
        padding_ref = torch.zeros((len(reference_points_per_img), self.max_length, 4), dtype=torch.float,
                                  device=self.level_embeds.device)
        for i, points in enumerate(reference_points_per_img):
            padding_ref[i, :points.shape[0], :points.shape[1]] = torch.from_numpy(points[:self.max_length]).to(
                self.level_embeds.device)
        return padding_ref

    def padding_along_img(self, reference_points, targets_words):
        num_ins = []
        max_num_ins = 0
        for i, reference_points_per_img in enumerate(reference_points):
            num_ins.append(reference_points_per_img.shape[0])
            max_num_ins = max(max_num_ins, reference_points_per_img.shape[0])
        padding_reference_points = torch.zeros(len(num_ins), max_num_ins, self.max_length, 4, dtype=torch.float,
                                               device=self.level_embeds.device)
        padding_target_words = torch.ones(len(num_ins), max_num_ins, self.max_length, dtype=torch.long,
                                          device=self.level_embeds.device) * self.convertor.padding_idx
        for i in range(len(reference_points)):
            padding_reference_points[i, :num_ins[i]] = reference_points[i]
            padding_target_words[i, :num_ins[i]] = targets_words[i]['padded_targets']
        return padding_reference_points, padding_target_words, num_ins

    def convert_targets(self, reference_points, target_words, img_metas):
        input_img_h, input_img_w = img_metas[0]['pad_shape'][:2]
        if not self.training and target_words is None:
            assert reference_points is not None
            target_words = [DC([""] * p.shape[0]) for p in reference_points]
            # padding_reference_point = self
        padding_targets_words = []
        padding_reference_points = []
        for i in range(len(target_words)):
            padding_targets_words.append(self.convertor.str2tensor(target_words[i].data))
            padding_reference_points.append(self.padding_points(reference_points[i]))
        padding_reference_points, padding_target_words, num_ins = self.padding_along_img(padding_reference_points,
                                                                                         padding_targets_words)

        if not self.with_poly:
            padding_reference_points = padding_reference_points[..., :2]
        else:
            padding_reference_points[..., 2] /= input_img_w
            padding_reference_points[..., 3] /= input_img_h

        padding_reference_points[..., 0] /= input_img_w
        padding_reference_points[..., 1] /= input_img_h

        return padding_reference_points, padding_target_words, num_ins

    def forward(self, mlvl_feats, mlvl_masks, mlvl_positional_encodings, padding_reference_points, padding_target_words,
                num_ins, gt_reference_points=None, gt_texts=None, img_metas=None, **kwargs):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            reference_points (list[list[Tensor(t x 2)]])
            target_words (list(list(str))
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['pad_shape'][:2]

        bak_padding_reference_points = padding_reference_points.clone()  # bug

        if self.training:
            decode_out, decode_reference_points = self.decoder(
                mlvl_feats,
                mlvl_masks,
                # query_embeds,
                # padding_reference_points[:,:,:,:2].clone(),
                padding_reference_points[:, :, :, :2],
                padding_target_words,
                mlvl_positional_encodings,
                reg_branches=None,  # noqa:E501
                cls_branches=None,  # noqa:E501
                img_metas=img_metas,
                num_ins=num_ins
            )

            delta_reference_points = self.reg_branches(decode_out)
            gt_padding_reference_points = torch.cat(
                [bak_padding_reference_points[i, :num_ins[i]] for i in range(len(num_ins))])

            shifted_reference_points, out_reference_points = self.points_step(padding_reference_points,
                                                                              delta_reference_points, num_ins,
                                                                              padding_reference_points)

            texts = self.classifier(decode_out)
            losses = self.loss(texts, out_reference_points, padding_target_words, gt_padding_reference_points, num_ins,
                               input_img_h, input_img_w)
            return losses, texts, shifted_reference_points
        else:
            assert batch_size == 1
            text_logits, reference_points, step_weights, step_sampling_grids = self.decoder(
                mlvl_feats,
                mlvl_masks,
                # query_embeds,
                padding_reference_points,
                padding_target_words,
                mlvl_positional_encodings,
                reg_branches=self.reg_branches,  # noqa:E501
                cls_branches=self.classifier,  # noqa:E501
                img_metas=img_metas
            )
            # reference_points = torch.cat((padding_reference_points[:,:,:1], reference_points), dim=2)
            texts_str, text_scores, out_reference_points, polygons = self.get_results(text_logits, reference_points,
                                                                                      img_metas)
            return texts_str, text_scores, out_reference_points, polygons, step_weights, step_sampling_grids

    def points_step(self, padding_reference_points, delta_reference_points, num_ins, ref=None):
        shifted_reference_points = torch.zeros_like(ref)
        shifted_reference_points[..., :2] = inverse_sigmoid(padding_reference_points[..., :2])
        delta_reference_points = torch.split(delta_reference_points, num_ins)
        out_reference_points = []
        for i in range(len(num_ins)):
            shifted_reference_points[i, :num_ins[i], :, :2] = delta_reference_points[i][:, :, :2] + inverse_sigmoid(
                padding_reference_points[i,
                :num_ins[i], :, :2])
            if self.with_poly:
                shifted_reference_points[i, :num_ins[i], :, 2:] = delta_reference_points[i][:, :,
                                                                  2:] + shifted_reference_points[i, :num_ins[i], :, :2]
            out_reference_points.append(shifted_reference_points[i, :num_ins[i]])
        shifted_reference_points[..., :] = shifted_reference_points[..., :].sigmoid()
        out_reference_points = torch.cat(out_reference_points)

        return shifted_reference_points, out_reference_points

    def get_results(self, text_logits, reference_points, img_metas):
        input_img_h, input_img_w = img_metas[0]['pad_shape'][:2]
        scale_factor = img_metas[0]['scale_factor'][:2]
        texts_idx, text_scores = self.convertor.tensor2idx(text_logits.squeeze(0))
        texts_str = self.convertor.idx2str(texts_idx)
        texts_str = [t.replace("<UKN>", u'\u53e3') for t in texts_str]
        reference_points[..., 0] *= input_img_w
        reference_points[..., 1] *= input_img_h
        if self.with_poly:
            reference_points[..., 2] *= input_img_w
            reference_points[..., 3] *= input_img_h

            top_p = reference_points[..., 2:]
            top_p = torch.cat(
                [top_p[:, :, 1:2] - reference_points[:, :, 1:2, :2] + reference_points[:, :, 0:1, :2], top_p[:, :, 1:]],
                dim=2)
            bot_p = reference_points[..., :2] - (top_p - reference_points[..., :2])

            top_p[..., 0] /= scale_factor[0]
            top_p[..., 1] /= scale_factor[1]
            bot_p[..., 0] /= scale_factor[0]
            bot_p[..., 1] /= scale_factor[1]
            polygons = []

        reference_points[..., 0] /= scale_factor[0]
        reference_points[..., 1] /= scale_factor[1]
        reference_points = reference_points.squeeze(0)
        out_reference_points = []
        for i in range(len(texts_str)):
            out_reference_points.append(reference_points[i, :len(texts_str[i]) + 2].cpu().numpy())
            if self.with_poly:
                tt = top_p[0, i, :len(texts_str[i]) + 2].cpu().numpy()
                bb = bot_p[0, i, :len(texts_str[i]) + 2].cpu().numpy()
                polygons.append(np.concatenate([tt, bb[::-1]], axis=0))
        if self.with_poly:
            return texts_str, text_scores, out_reference_points, polygons
        return texts_str, text_scores, out_reference_points, None

    # @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             pred_text_logits,
             pred_reference_points,
             gt_texts,
             gt_reference_points,
             num_ins,
             input_image_h,
             input_image_w, ):
        """"Loss function.

        Args:
            pred_text_logits: N x n x t x num_classes
            pred_reference_points: N x n x t x 2
            gt_texts: N x n x t
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        gt_texts_real = []
        for i in range(len(num_ins)):
            gt_texts_real.append(gt_texts[i, :num_ins[i]])
        gt_texts_real = torch.cat(gt_texts_real, dim=0)
        pred_texts_real = pred_text_logits
        pred_reference_points_real = pred_reference_points
        gt_reference_points_targets_real = gt_reference_points

        mask = (gt_texts_real != self.convertor.padding_idx).float()[:, 1:].contiguous()

        gt_texts_real = gt_texts_real[:, 1:].contiguous()
        gt_reference_points_targets_real = gt_reference_points_targets_real[:, 1:].contiguous()

        pred_texts_real = pred_texts_real[:, :-1].contiguous()
        pred_reference_points_real = pred_reference_points_real[:, :-1].contiguous()

        losses = {}

        losses['loss_ce_text'] = self.loss_texts(pred_texts_real, gt_texts_real, mask, pred_texts_real.shape[0]) * 2

        losses['loss_l1_points'] = self.loss_reference_points(pred_reference_points_real,
                                                              gt_reference_points_targets_real, mask)
        de = losses['loss_ce_text'].detach().cpu().numpy()
        if not np.isfinite(de):
            raise ValueError

        return losses

    def loss_texts(self, pred_texts, text_targets, mask, num_ins):
        return (F.cross_entropy(pred_texts.view(-1, self.num_classes), text_targets.reshape(-1).long(),
                                reduction='none') * mask.reshape(-1)).sum() / (mask.sum() + 1e-4)

    def loss_reference_points(self, pred_points, target_points, mask):
        return (F.smooth_l1_loss(pred_points * 100, target_points * 100, reduction="none").sum(dim=-1) * mask).sum() / (
                    mask.sum() + 1e-4)