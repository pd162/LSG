import torch
import torch.nn.functional as F
from mmdet.core import BitmapMasks
# from mmocr.models.builder import LOSSES
from mmdet.models.builder import LOSSES
from torch import nn
from mmdet.core import multi_apply
from mmocr.utils import check_argument


@LOSSES.register_module()
class LSGDetLoss(nn.Module):
    """The class for implementing TextSnake loss:
    TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes
    [https://arxiv.org/abs/1807.01544].
    This is partially adapted from
    https://github.com/princewang1994/TextSnake.pytorch.
    """

    def __init__(
            self,
            ohem_ratio=3.0,
            with_head=True,
            loss_text_weight=1.,
            loss_head_weight=1.,
            loss_center_weight=1.,
    ):
        """Initialization.

        Args:
            ohem_ratio (float): The negative/positive ratio in ohem.
        """
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.with_head = with_head
        self.loss_text_weight = loss_text_weight
        self.loss_head_weight = loss_head_weight
        self.loss_center_weight = loss_center_weight

    def balanced_bce_loss(self, pred, gt, mask):

        assert pred.shape == gt.shape == mask.shape
        try:
            assert not (torch.isnan(pred).all().item() or torch.isnan(gt).all().item() or torch.isnan(mask).all().item())  # nan
            # assert torch.isnan(pred)
            positive = gt * mask
            negative = (1 - gt) * mask
            positive_count = int(positive.float().sum())
            gt = gt.float()
            if positive_count > 0:
                loss = F.binary_cross_entropy(pred, gt, reduction='none')  # bug
                # loss = self.bce_loss(pred, gt)
                positive_loss = torch.sum(loss * positive.float())
                negative_loss = loss * negative.float()
                negative_count = min(
                    int(negative.float().sum()),
                    int(positive_count * self.ohem_ratio))
            else:
                positive_loss = torch.tensor(0.0, device=pred.device)
                loss = F.binary_cross_entropy(pred, gt, reduction='none')
                negative_loss = loss * negative.float()
                negative_count = 100
            negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

            balance_loss = (positive_loss + torch.sum(negative_loss)) / (
                    float(positive_count + negative_count) + 1e-5)
        except:
            device = pred.device
            temp_out = torch.tensor(0.).to(device)
            temp_out.requires_grad = True
            balance_loss = temp_out

        return balance_loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor size HxW.

        Returns
            results (list[tensor]): The list of kernel tensors. Each
                element is for one kernel level.
        """
        assert check_argument.is_type_list(bitmasks, BitmapMasks)
        assert isinstance(target_sz, tuple)

        batch_size = len(bitmasks)
        num_masks = len(bitmasks[0])

        results = []

        for level_inx in range(num_masks):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                # hxw
                mask_sz = mask.shape
                # left, right, top, bottom
                pad = [
                    0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
                ]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            results.append(kernel)

        return results

    def forward(self, pred_maps, downsample_ratio, gt_text_mask=None,
                gt_head_mask=None, gt_center_mask=None, gt_mask=None, **kwargs):
        if self.with_head:
            loss_text, loss_center, loss_head = self.forward_single(pred_maps, gt_text_mask, gt_mask,
                                                                    gt_head_mask, gt_center_mask)
            return dict(loss_text=loss_text,
                        loss_center=loss_center,
                        loss_head=loss_head)
        else:
            gt_text_masks = []
            gt_masks = []
            gt_text_masks.append(gt_text_mask)
            gt_masks.append(gt_mask)
            _, l0_h, l0_w = gt_text_mask.shape
            for i in range(1, len(pred_maps)):
                lvl_gt_text_mask = F.interpolate(gt_text_mask.unsqueeze(1), (l0_h // (2 ** i), l0_w // (2 ** i)))
                lvl_gt_text_mask = (lvl_gt_text_mask > 0.2).to(torch.int64).squeeze(1)

                gt_text_masks.append(lvl_gt_text_mask)
                if gt_mask is not None:
                    lvl_gt_mask = F.interpolate(gt_mask.unsqueeze(1), (l0_h // (2 ** i), l0_w // (2 ** i)))
                    lvl_gt_mask = (lvl_gt_mask > 0.2).to(torch.int64).squeeze(1)
                else:
                    lvl_gt_mask = None

                gt_masks.append(lvl_gt_mask)

            losses = multi_apply(self.forward_single, pred_maps, gt_text_masks, gt_masks)
            loss_text = sum(losses[0])
            return dict(loss_text=loss_text)

    def forward_single(self, pred_map, gt_text_mask=None, gt_mask=None,
                       gt_head_mask=None, gt_center_mask=None,
                       gt_shape=None, gt_color=None):

        pred_map = pred_map[0]
        pred_text_region = pred_map[0][:, 0, :, :]
        if self.with_head:
            pred_head_region = pred_map[0][:, 1, :, :]
            pred_center_region = pred_map[0][:, 2, :, :]
        feature_sz = pred_map[0].size()
        device = pred_map[0].device

        if gt_mask is None:
            gt_mask = torch.ones_like(gt_text_mask)

        loss_text = self.balanced_bce_loss(
            torch.sigmoid(pred_text_region), gt_text_mask.to(pred_text_region.dtype), gt_mask)  # whole map

        loss_center = torch.tensor(0.).to(pred_text_region.dtype).to(pred_text_region.device)
        loss_head = torch.tensor(0.).to(pred_text_region.dtype).to(pred_text_region.device)
        loss_center.requires_grad = True
        loss_head.requires_grad = True
        if self.with_head:
            text_mask = (gt_text_mask * gt_mask).to(pred_text_region.dtype)
            loss_center_map = F.binary_cross_entropy(
                torch.sigmoid(pred_center_region),
                gt_center_mask.to(pred_text_region.dtype),
                reduction='none')

            if int(text_mask.sum()) > 0:
                loss_center = torch.sum(
                    loss_center_map * text_mask) / torch.sum(text_mask)
            else:
                loss_center = torch.tensor(0.0, device=device)

            if int(gt_center_mask.sum()) > 0:
                # map_sz = pred_head_region.size()
                # ones = torch.ones(map_sz, dtype=pred_head_region.dtype, device=device)
                loss_head = torch.sum(
                    F.smooth_l1_loss(torch.sigmoid(pred_head_region), gt_head_mask.to(pred_head_region.dtype),
                                     reduction='none') \
                    * gt_center_mask) / torch.sum(gt_center_mask)

                # loss_head = F.smooth_l1_loss(torch.sigmoid(pred_head_region), gt_head_mask.to(pred_head_region.dtype),
                #                      reduction='none')

            # loss_shape = nn.CrossEntropyLoss()(pred_shape, gt_shape)
            # loss_color = nn.CrossEntropyLoss()(pred_color, gt_color)

        if self.with_head:
            return self.loss_text_weight * loss_text, self.loss_center_weight *  loss_center, self.loss_head_weight *  loss_head

        return self.loss_text_weight * loss_text,



class RAHReadLoss(nn.Module):

    def __init__(self, embed_dims, num_classes):
        super(RAHReadLoss, self).__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.mlvl_fuse = nn.Linear(self.embed_dims * 4, self.embed_dims)
        self.classifier = nn.Linear(self.embed_dims, self.num_classes)

    def grid_sample(self, feat, sampling_points):
        N, n, t, p, _ = sampling_points.shape
        # if feat_pos_embedding is not None and self.add_pos:
        #     feat = feat + feat_pos_embedding
        feat = feat.detach()
        sampling_points = sampling_points.reshape(N, n, -1, 2) * 2 - 1
        sampling_feature = F.grid_sample(feat, sampling_points, mode="bilinear", align_corners=True)  # N C n t*p
        return (sampling_feature.view(N, -1, n, t, p),)

    def sample_features(self, mlvl_feats, mlvl_sampling_points):
        '''

        :param mlvl_feats: (list[Tensor N x C x H x W])
        :param sampling_points: list[N x n x t x p x 2]
        :return:
        '''
        if len(mlvl_sampling_points) == 1:
            mlvl_sampling_points = mlvl_sampling_points * len(mlvl_feats)
            # mlvl_pos_embeds =mlvl_pos_embeds * len(mlvl_feats)
        mlvl_sampling_features = multi_apply(self.grid_sample, list(mlvl_feats), mlvl_sampling_points)[0]
        mlvl_sampling_features = torch.cat(mlvl_sampling_features, dim=-1)  # N C n t p*nl
        mlvl_sampling_features = mlvl_sampling_features.permute(0, 2, 3, 4, 1).contiguous()  # N n t p*nl C
        return mlvl_sampling_features

    def forward(self, mlvl_feats,
                # mlvl_masks,
                padding_reference_points,
                # padding_target_words,
                # mlvl_pos_embeds, img_metas, num_ins=None
                num_ins
                ):
        # input_img_h, input_img_w = img_metas[0]['pad_shape'][:2]
        # N, n, t = padding_target_words.shape
        # targets_embedding = self.trg_word_emb(padding_target_words)  # N x n x t x C

        shifted_points = padding_reference_points.unsqueeze(-2)
        shifted_points_feature = self.sample_features(mlvl_feats, [shifted_points])  # N x n x t x p*nl x C

        shifted_points_feature = shifted_points_feature.flatten(-2)
        shifted_points_feature = torch.cat([shifted_points_feature[i, :num_ins[i]] for i in range(len(num_ins))])
        shifted_points_feature = self.mlvl_fuse(shifted_points_feature)

        decode_logits = self.classifier(shifted_points_feature)
        return decode_logits
