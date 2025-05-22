import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.runner import ModuleList
from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.decoders import BaseDecoder
#
from mmocr.models.common.modules import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    len_s = seq.size(1)
    subsequent_mask = 1 - torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).bool()
    return subsequent_mask


@DECODERS.register_module()
class AutoRegTF(BaseDecoder):

    def __init__(self,
                 grid_size=(5, 5),
                 n_level=4,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=93,
                 max_seq_len=40,
                 start_idx=1,
                 padding_idx=92,
                 disturb_range=1.0,
                 disturb_vert=False,
                 add_pos=False,
                 sample_mode="bilinear",
                 boxfix=False,
                 init_cfg=None,
                 sampling_range=[4, 8, 16, 32],
                 return_weights=False,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.d_model = d_model
        self.grid_size = grid_size
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len
        self.disturb_range = disturb_range
        self.add_pos = add_pos
        self.sample_mode = sample_mode
        self.disturb_vert = disturb_vert
        self.boxfix = boxfix
        self.return_weights = return_weights
        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)
        self.sampling_range = sampling_range

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = ModuleList([
            AutoRegTransformerDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.lvl_merge = nn.Linear(d_model, d_model)
        self.token_scale = nn.Linear(d_model, d_model)

        self.sample_scale = nn.Linear(d_model, 2)
        constant_init(self.sample_scale, 0, 0)

        y_range = torch.linspace(-0.5, 0.5, self.grid_size[0]).float()
        x_range = torch.linspace(-0.5, 0.5, self.grid_size[1]).float()
        yy, xx = torch.meshgrid(y_range, x_range)
        norm_grids = torch.stack((xx, yy), dim=-1).view(1, 1, 1, 1, self.grid_size[0] * self.grid_size[1], 2)
        # scale_ratio = torch.tensor([4, 8, 16, 32], dtype=torch.float).view(1, 1, 1, 4, 1, 1) * (grid_size[0] - 1) * 2
        if self.sampling_range[0] > 1:
            scale_ratio = torch.tensor(self.sampling_range, dtype=torch.float).view(1, 1, 1, 4, 1, 1) * (
                        grid_size[0] - 1) * 2
        else:
            scale_ratio = torch.tensor(self.sampling_range, dtype=torch.float).view(1, 1, 1, 4, 1, 1)
        # self.norm_grids = norm_grids * scale_ratio
        # self.register_buffer("norm_grids", norm_grids*scale_ratio)
        self.norm_grids = norm_grids * scale_ratio
        # scale_ratio = torch.tensor([4, 8, 16, 32], dtype=torch.float).view(1, 1,1, 4, 1,1) * (grid_size[0] - 1)
        # self.scale_ratio = nn.Parameter(scale_ratio, requires_grad=False)

    # def __init_weights

    def generate_sample_grids(self, sample_scales, reference_points, input_img_h, input_img_w, delta_hw=None):
        '''
        :param sample_scales: N, n, t, nl, 2
        :return:
        '''
        self.norm_grids = self.norm_grids.to(sample_scales.device)
        if delta_hw is not None:
            reference_points = (inverse_sigmoid(reference_points.unsqueeze(-2)) + delta_hw).sigmoid()
        else:
            reference_points = reference_points.unsqueeze(-2)
        sample_scales = torch.sigmoid(sample_scales)
        if self.sampling_range[0] > 1:
            sample_grids = sample_scales[:, :, :, :, None, :] * self.norm_grids / torch.tensor(
                [input_img_w, input_img_h], dtype=sample_scales.dtype, device=sample_scales.device).view(1, 1, 1, 1, 1,
                                                                                                         2)
        else:
            sample_grids = sample_scales[:, :, :, :, None, :] * self.norm_grids
        sample_grids = reference_points[:, :, :, :, None, :] + sample_grids  # N, n , t, nl, p, 2
        sample_grids = [sample_grids_per_level.squeeze(-3) for sample_grids_per_level in
                        torch.split(sample_grids, 1, dim=-3)]
        grids_mask = [(grids[..., 0] > 0) * (grids[..., 0] < 1) * (grids[..., 1] > 0) * (grids[..., 1] < 1) for grids in
                      sample_grids]
        return sample_grids, reference_points.mean(dim=-2), grids_mask

    def grid_sample(self, feat, feat_pos_embedding, sampling_points):
        N, n, t, p, _ = sampling_points.shape
        if feat_pos_embedding is not None and self.add_pos:
            feat = feat + feat_pos_embedding
        sampling_points = sampling_points.reshape(N, n, -1, 2) * 2 - 1
        sampling_feature = F.grid_sample(feat, sampling_points, mode=self.sample_mode)  # N C n t*p
        return (sampling_feature.view(N, -1, n, t, p),)

    def sample_features(self, mlvl_feats, mlvl_sampling_points, mlvl_pos_embeds):
        '''

        :param mlvl_feats: (list[Tensor N x C x H x W])
        :param sampling_points: list[N x n x t x p x 2]
        :return:
        '''
        if len(mlvl_sampling_points) == 1:
            mlvl_sampling_points = mlvl_sampling_points * len(mlvl_feats)
            # mlvl_pos_embeds =mlvl_pos_embeds * len(mlvl_feats)
        mlvl_sampling_features = multi_apply(self.grid_sample, list(mlvl_feats), mlvl_pos_embeds, mlvl_sampling_points)[
            0]
        mlvl_sampling_features = torch.cat(mlvl_sampling_features, dim=-1)  # N C n t p*nl
        mlvl_sampling_features = mlvl_sampling_features.permute(0, 2, 3, 4, 1).contiguous()  # N n t p*nl C
        return mlvl_sampling_features

    def _attention(self, trg_seq, trg_embedding, src, src_mask=None):
        # trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = get_pad_mask(
            trg_seq, pad_idx=self.padding_idx) & get_subsequent_mask(trg_seq)
        trg_mask[:, :, 0] = True
        # trg_mask = trg_mask == False
        output = tgt
        layer_weights = []
        for dec_layer in self.layer_stack:
            output, weights = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
            layer_weights.append(weights)
        output = self.layer_norm(output)

        return output, layer_weights

    def point_disturb(self, reference_points):
        # reference_points: N x 26 x 2
        zero_mask = reference_points == 0
        shift_dir = reference_points[:, :, 1:, :2] - reference_points[:, :, :-1, :2]  # N x 25 x 2
        shift_dir = torch.cat([shift_dir[:, :, :1], shift_dir], dim=2)
        shift_scale = torch.rand(shift_dir.shape[:3], dtype=shift_dir.dtype, device=shift_dir.device).unsqueeze(
            -1) * self.disturb_range - self.disturb_range / 2.
        reference_points[:, :, :, :2] = reference_points[:, :, :, :2] + shift_dir * shift_scale

        if self.disturb_vert:
            shift_scale_v = torch.rand(shift_dir.shape[:3], dtype=shift_dir.dtype, device=shift_dir.device).unsqueeze(
                -1) * self.disturb_range - self.disturb_range / 2.
            shift_dir_v = shift_dir.clone()

            shift_dir_v[..., 0] = shift_dir[..., 1]
            shift_dir_v[..., 1] = -shift_dir[..., 0]

            reference_points[:, :, :, :2] = reference_points[:, :, :, :2] + shift_dir_v * shift_scale_v
        reference_points[zero_mask] = 0
        reference_points = reference_points.clamp(min=0, max=1)
        return reference_points

    def forward(self, mlvl_feats,
                mlvl_masks,
                # query_embeds,
                padding_reference_points,
                padding_target_words,
                mlvl_pos_embeds,
                reg_branches,  # noqa:E501
                cls_branches,
                img_metas,
                num_ins=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.
            padding_reference_points (Tensor N x n x t x 2)
            padding_target_words (Tensor N x n x t)
        Returns:
        """

        # padding_reference_points = self.point_disturb(padding_reference_points)
        #
        N, n, t = padding_target_words.shape
        if self.training:
            padding_reference_points = self.point_disturb(padding_reference_points)
            padding_target_words, targets_embedding, grids_features, grids_mask, padding_reference_points, sampling_grids = self.forward_step(
                mlvl_feats,
                mlvl_masks,
                # query_embeds,
                padding_reference_points,
                padding_target_words,
                mlvl_pos_embeds, img_metas, num_ins)

            decoder_out, _ = self._attention(padding_target_words, targets_embedding, grids_features,
                                             src_mask=grids_mask)

            return decoder_out.view(-1, t, self.d_model), padding_reference_points
        else:
            # padding_reference_points = self.point_disturb(padding_reference_points)
            return self.forward_test(mlvl_feats,
                                     mlvl_masks,
                                     # query_embeds,
                                     padding_reference_points,
                                     padding_target_words,
                                     mlvl_pos_embeds,
                                     reg_branches,  # noqa:E501
                                     cls_branches,
                                     img_metas)

    def forward_step(self, mlvl_feats,
                     mlvl_masks,
                     padding_reference_points,
                     padding_target_words,
                     mlvl_pos_embeds, img_metas, num_ins=None):

        input_img_h, input_img_w = img_metas[0]['pad_shape'][:2]
        N, n, t = padding_target_words.shape
        targets_embedding = self.trg_word_emb(padding_target_words)  # N x n x t x C

        init_points = padding_reference_points.unsqueeze(-2)
        init_points_feature = self.sample_features(mlvl_feats, [init_points], mlvl_pos_embeds)  # N x n x t x p*nl x C

        init_point_text_feature = self.lvl_merge(init_points_feature) + self.token_scale(targets_embedding).unsqueeze(
            -2)

        grid_h_w = self.sample_scale(init_point_text_feature)  # N x n x t x nl x 2
        delta_hw = None
        if self.boxfix:
            grid_h_w = grid_h_w * 0.0
        sampling_grids, padding_reference_points, grids_mask = self.generate_sample_grids(grid_h_w,
                                                                                          padding_reference_points,
                                                                                          input_img_h, input_img_w,
                                                                                          delta_hw)

        grids_features = self.sample_features(mlvl_feats, sampling_grids, mlvl_pos_embeds)  # N x n x t x p*nl x C
        grids_mask = torch.cat(grids_mask, dim=-1)

        if num_ins is not None:
            targets_embedding = torch.cat([targets_embedding[i, :num_ins[i]] for i in range(len(num_ins))], dim=0)
            padding_target_words = torch.cat([padding_target_words[i, :num_ins[i]] for i in range(len(num_ins))], dim=0)
            grids_features = torch.cat([grids_features[i, :num_ins[i]] for i in range(len(num_ins))], dim=0)
            grids_mask = torch.cat([grids_mask[i, :num_ins[i]] for i in range(len(num_ins))], dim=0)
        else:
            targets_embedding = targets_embedding.view(N * n, t, self.d_model)
            padding_target_words = padding_target_words.view(N * n, t)
            grids_features = grids_features.view(N * n, t, -1, self.d_model)
            grids_mask = grids_mask.view(N * n, t, -1)
        if (grids_mask.sum(-1) == 0).sum() > 0:
            # print(grids_mask)
            grids_mask[:, :, 0][grids_mask.sum(-1) == 0] = True
        if not self.return_weights:
            sampling_grids = None
        # decoder_out = self._attention(padding_target_words, targets_embedding, grids_features)
        return padding_target_words, targets_embedding, grids_features, grids_mask, padding_reference_points, sampling_grids

    def forward_test(self, mlvl_feats,
                     mlvl_masks,
                     # query_embeds,
                     padding_reference_points,
                     padding_target_words,
                     mlvl_pos_embeds,
                     reg_branches,  # noqa:E501
                     cls_branches,
                     img_metas):
        N, n, t = padding_target_words.shape
        output_texts = []
        step_weights = []
        step_sampling_grids = []
        grids_features = None
        targets_embedding = None
        grids_mask = None
        # padding_reference_points = self.point_disturb(padding_reference_points)
        finish_label = torch.zeros(N * n, dtype=torch.bool, device=padding_target_words.device)
        for step in range(self.max_seq_len):
            step_padding_target_words, step_targets_embedding, step_grids_features, step_grids_mask, steped_reference_points, sampling_grids = self.forward_step(
                mlvl_feats,
                mlvl_masks,
                padding_reference_points[:, :, step:step + 1, :2],
                padding_target_words[:, :, step:step + 1],
                mlvl_pos_embeds,
                img_metas)
            if grids_features is None:
                grids_features = step_grids_features
                targets_embedding = step_targets_embedding
                grids_mask = step_grids_mask
            else:
                grids_features = torch.cat([grids_features, step_grids_features], dim=1)
                grids_mask = torch.cat([grids_mask, step_grids_mask], dim=1)
                targets_embedding = torch.cat([targets_embedding, step_targets_embedding], dim=1)

            decode_out, weights = self._attention(padding_target_words[:, :, :step + 1].contiguous().view(N * n, -1),
                                                  targets_embedding, grids_features, src_mask=grids_mask)
            decode_out = decode_out.view(N, n, step + 1, self.d_model)
            step_weights.append(weights)
            step_sampling_grids.append(sampling_grids)
            new_reference_points = reg_branches(decode_out[:, :, step])

            new_reference_points[..., :2] = new_reference_points[..., :2] + inverse_sigmoid(
                padding_reference_points[:, :, step, :2])
            new_reference_points[..., :2] = new_reference_points[..., :2].sigmoid()
            if new_reference_points.shape[-1] == 4:
                new_reference_points[..., 2:] = (
                            inverse_sigmoid(new_reference_points[..., :2]) + new_reference_points[..., 2:]).sigmoid()

            step_text_logit = F.softmax(cls_branches(decode_out[:, :, step]), dim=-1)
            output_texts.append(step_text_logit)
            _, step_text_index = torch.max(step_text_logit, dim=-1)

            if step < self.max_seq_len - 1:
                padding_reference_points[:, :, step + 1, :] = new_reference_points[..., :]
                padding_target_words[:, :, step + 1] = step_text_index

            finish_label[step_text_index[0] == self.start_idx] = True
            if finish_label.sum() == N * n:
                break

        output_texts = torch.stack(output_texts, dim=2)

        return output_texts, padding_reference_points, step_weights, step_sampling_grids


class AutoRegTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=1,
                 return_weights=False,
                 act_cfg=dict(type='mmcv.GELU')):
        super().__init__()
        self.mask_value = mask_value
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.return_weights = return_weights
        self.self_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias, )
        self.enc_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        '''
        :param dec_input: N t c
        :param enc_output: N t p c
        :param self_attn_mask:
        :param dec_enc_attn_mask:
        :return:
        '''
        self_attn_in = self.norm1(dec_input)
        self_attn_out = self.self_attn(self_attn_in, self_attn_in,
                                       self_attn_in, self_attn_mask == self.mask_value)
        enc_attn_in = dec_input + self_attn_out

        enc_attn_q = self.norm2(enc_attn_in)
        N, t, C = enc_attn_q.shape
        enc_attn_q = enc_attn_q.view(N * t, 1, C)
        enc_output = enc_output.view(N * t, -1, C)
        dec_enc_attn_mask = dec_enc_attn_mask.view(N * t, -1)
        enc_attn_out = self.enc_attn(enc_attn_q, enc_output, enc_output,
                                     dec_enc_attn_mask == self.mask_value)  # N*t, 1, c
        weights = None
        enc_attn_out = enc_attn_out.view(N, t, C)
        # enc_attn_out = torch.nan_to_num(enc_attn_out,0)
        mlp_in = enc_attn_in + enc_attn_out
        mlp_out = self.mlp(self.norm3(mlp_in))
        out = mlp_in + mlp_out

        return out, weights