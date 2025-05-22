import torch
import torch.nn.functional as F
import mmcv
from PIL import Image, ImageDraw, ImageFont
import warnings
import cv2
import random
import numpy as np
from torch import nn
from mmdet.models.builder import build_head
from mmocr.models.builder import DETECTORS
from mmocr.models.textdet.detectors import SingleStageTextDetector
from mmdet.core import multi_apply


@DETECTORS.register_module()
class LSG(SingleStageTextDetector):

    def __init__(self, backbone,
                 neck,
                 bbox_head,
                 recog_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None):
        super(LSG, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg)
        self.recog_head = build_head(recog_head)
        self.dd = nn.Conv2d(3, 3, (1, 1))

    def mlvl_feats_mask_pos_encoding(self, mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['pad_shape'][:2]
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for lvl, feat in enumerate(mlvl_feats):
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.recog_head.positional_encoding(mlvl_masks[-1]) + self.recog_head.level_embeds[lvl].view(1, -1, 1,
                                                                                                             1)
            )
        return mlvl_positional_encodings, mlvl_masks

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # try:
        x = self.extract_feat(img)
        kwargs['img'] = img
        mlvl_positional_encodings, mlvl_masks = self.mlvl_feats_mask_pos_encoding(x, img_metas)

        padding_reference_points, padding_target_words, num_ins = self.recog_head.convert_targets(
            kwargs['gt_reference_points'], kwargs['gt_texts'], img_metas)  # tokenizer
        # assert not torch.isnan(x[0]).any().item()
        det_maps = self.bbox_head(x[0])
        det_losses = self.bbox_head.loss(det_maps, **kwargs)  # LSGDetLoss bug: cuda-trigger
        recog_losses, _, _ = self.recog_head(x, mlvl_masks, mlvl_positional_encodings, padding_reference_points,
                                             padding_target_words, num_ins, img_metas=img_metas,
                                             **kwargs)
        # except Exception as e:
        #     print(e)
        #     print(img_metas[0]['ori_filename'], img_metas[1]['ori_filename'],
        #           img_metas[2]['ori_filename'], img_metas[3]['ori_filename'])
        #     # out = self.dd(img[:, :, :10, :10]).sum() * 0  # 越界之后无法使用img tensor
        #     device = img.device
        #     temp_out = torch.tensor(0.).to(device)
        #     temp_out.requires_grad = True
        #     recog_losses = {
        #         'loss_ce_text': temp_out,
        #         'loss_l1_points': temp_out,
        #     }
        #     det_losses = {
        #         'loss_text': temp_out,
        #         'loss_head': temp_out,
        #         'loss_center': temp_out,
        #     }
        #     torch.cuda.empty_cache()
        losses = {}
        losses.update(det_losses)
        losses.update(recog_losses)
        return losses

    def simple_test(self, img, img_metas, **kwargs):
        # pass
        # try:
            x = self.extract_feat(img)
            det_res = self.bbox_head(x[0])
            det_maps = det_res[0]
            # shape = torch.argmax(det_res[1])
            # color = torch.argmax(det_res[2])
            det_results = self.bbox_head.get_start_points(det_maps)
            # det_results = dict(
            #     reference_points=det_res[0],
            #     reference_points_score=det_res[1]
            # )
            mlvl_positional_encodings, mlvl_masks = self.mlvl_feats_mask_pos_encoding(x, img_metas)
            polygons = None
            if getattr(self.test_cfg, "use_gt", False):
                det_results = {}
                kwargs['gt_texts'] = kwargs['gt_texts'][0]
                kwargs['gt_reference_points'] = kwargs['gt_reference_points'][0]
                # temp_list = []
                # for item in kwargs['gt_reference_points'][0]:
                #     temp_list.append(item[0, :2].reshape(1, 2))
                # kwargs['gt_reference_points'] = [np.array(temp_list)]

                if len(kwargs['gt_reference_points'][0]) == 0:
                    texts_str = []
                    text_scores = []
                    out_reference_points = []
                    polygons = []
                    step_weights = []
                    step_sampling_grids = []
                    det_scores = np.array([])
                else:
                    padding_reference_points, padding_target_words, num_ins = self.recog_head.convert_targets(
                        kwargs['gt_reference_points'], kwargs['gt_texts'], img_metas)

                    texts_str, text_scores, out_reference_points, polygons, step_weights, step_sampling_grids = self.recog_head(
                        x, mlvl_masks, mlvl_positional_encodings, padding_reference_points, padding_target_words, num_ins,
                        img_metas=img_metas, **kwargs)
                    det_results['reference_points_score'] = torch.tensor([], device=x[0].device)
                    det_scores = det_results['reference_points_score'].cpu().numpy()
            else:
                kwargs['gt_reference_points'] = [det_results['reference_points']]
                kwargs['gt_texts'] = None
                if det_results['reference_points'].shape[0] == 0:
                    texts_str = []
                    text_scores = []
                    out_reference_points = []
                    polygons = []
                    step_weights = []
                    step_sampling_grids = []
                else:
                    # grid_x, grid_y = np.meshgrid(np.arange(1, 101), np.arange(1, 101))
                    # grid = np.stack((grid_x, grid_y), axis=-1)
                    # grid = grid.reshape((10000, 1, 2))

                    padding_reference_points, padding_target_words, num_ins = self.recog_head.convert_targets(
                        kwargs['gt_reference_points'], kwargs['gt_texts'], img_metas)
                    # padding_reference_points, padding_target_words, num_ins = self.recog_head.convert_targets(
                    #     grid, kwargs['gt_texts'], img_metas)
                    texts_str, text_scores, out_reference_points, polygons, step_weights, step_sampling_grids = self.recog_head(
                        x, mlvl_masks, mlvl_positional_encodings, padding_reference_points, padding_target_words, num_ins,
                        img_metas=img_metas, **kwargs)
                    texts_str = [text.replace("口", ' ') for text in texts_str]

            return [{
                "strs": texts_str,
                "char_scores": text_scores,
                # "det_scores": det_results['reference_points_score'].cpu().numpy(),
                "det_scores": det_scores,
                "reference_points": out_reference_points,
                "polygons": polygons,
                "step_weights": step_weights,
                "step_sampling_grids": step_sampling_grids,
                "img_metas": img_metas,
                # 'shape': shape,
                # 'color': color
            }]
        # except Exception as e:
        #     print(e)
        #     return [{
        #         "strs": [],
        #         "char_scores": [],
        #         "det_scores": 0.,
        #         "reference_points": [],
        #         "polygons": [],
        #         "step_weights": [],
        #         "step_sampling_grids": [],
        #         "img_metas": img_metas,
        #         # 'shape': shape,
        #         # 'color': color
        #     }]

    def cv2ImgAddText(self, img, text, left, top, textColor, textSize):
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        ttf = "../PaddleOCR/StyleText/fonts/ch_standard.ttf"
        fontStyle = ImageFont.truetype(
            ttf, textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def show_result(self,
                    img,
                    result,
                    score_thr=0.1,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img`.
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.imshow_pred_boundary`
        """
        # try:
        img = mmcv.imread(img)
        img = img.copy()
        h, w = img.shape[:2]
        scale = max(h, w) // 200
        points = result['reference_points']
        strs = result['strs']
        char_scores = result['char_scores']
        det_scores = result['det_scores']
        polygons = result['polygons']
        step_weights = result['step_weights']
        step_sampling_grids = result['step_sampling_grids']
        for i in range(len(points)):
            score_det = det_scores[i]
            score = np.min(char_scores[i])  # 过滤一些错误的实例
            # if score * score_det < score_thr:
            #     continue
            color = (255, 0, 0)
            center_points = points[i][:, :2].reshape(-1, 2).astype('int')
            for point in center_points:
                cv2.circle(img, point.tolist(), scale, color, scale)

            img = self.cv2ImgAddText(img, strs[i].replace("口", ' '), center_points[0][0], center_points[0][1], color, scale * 5 + 10)

        # img = self.cv2ImgAddText(img, color_dict[color_idx], 10, 10, (255, 0, 0), scale*5+10)
        # img = self.cv2ImgAddText(img, shape_dict[shape_idx], 30, 10, (255, 0, 0), scale*5+10)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
        return img
        # except Exception as e:
        #     print(e)
        #     return img