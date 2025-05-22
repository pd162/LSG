import os

from .base_textdet_targets import BaseTextDetTargets
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
import mmocr.utils.check_argument as check_argument
from scipy.stats import multivariate_normal
from .tpsnet_targets import TPSTargets


@PIPELINES.register_module()
class LSGTargets(TPSTargets):

    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            assert len(poly) == 1
            polygon_points = poly[0].reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = 0
            line_tail_shrink_len = 0

            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                                     head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                                     head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                        resampled_top_line[i + 1] -
                        center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                        resampled_bot_line[i + 1] -
                        center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def generate_text_region_mask(self, img_size, text_polys, downsample_rate=4):

        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(
                np.round(text_instance), dtype=np.int32).reshape((1, -1, 2))

            cv2.fillPoly(text_region_mask, polygon, 1)
        text_region_mask = cv2.resize(text_region_mask, (w // downsample_rate, h // downsample_rate))
        return text_region_mask

    def generate_start_map(self, img_size, reference_points, polygons, downsample_rate=4.0):

        h, w = img_size
        text_start_mask = np.zeros((h, w), dtype=np.float32)
        XX, YY = np.meshgrid(np.arange(w), np.arange(h))
        # Z = clf.pdf(np.dstack([XX, YY])).reshape(div, div)
        # mask = Z / Z.max()
        center_region_masks = [self.generate_center_region_mask(img_size, [polygons[i]]) for i in range(len(polygons))]
        for i, reference_point in enumerate(reference_points):
            reference_point = reference_point[:, :2]
            start_point = reference_point[0]

            start_dis = np.linalg.norm(reference_point[len(reference_point[len(reference_point) // 2])] - start_point,
                                       ord=2)
            clf = multivariate_normal(mean=start_point,
                                      cov=[[start_dis ** 2, 0], [0, start_dis ** 2]],
                                      allow_singular=True
                                      )  # bug
            start_mask_per_ins = clf.pdf(np.dstack([XX, YY])).reshape(h, w)
            start_mask_per_ins = start_mask_per_ins / (start_mask_per_ins.max() + start_mask_per_ins.min() + 1e-6)  # invalid value
            start_mask_per_ins = start_mask_per_ins * center_region_masks[i]
            text_start_mask += start_mask_per_ins

        text_start_mask = cv2.resize(text_start_mask, (int(w // downsample_rate), int(h // downsample_rate)))
        # text_start_mask = text_start_mask/(text_start_mask.max() + 1e-4)
        center_region_mask = np.zeros((h, w))
        for mask in center_region_masks:
            center_region_mask += mask
        center_region_mask = cv2.resize(center_region_mask, (int(w // downsample_rate), int(h // downsample_rate)))
        return text_start_mask, center_region_mask

    def resample_polygon(self, top_line, bot_line, n=None):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        resample_line = []
        for polygon in [top_line, bot_line]:
            x, y = polygon[:, 0], polygon[:, 1]
            tck, u = splprep([x, y], k=3 if polygon.shape[0] >= 5 else 2 if polygon.shape[0] > 2 else 1, s=0)  # bug
            u = np.linspace(0, 1, num=n, endpoint=True)
            out = splev(u, tck)
            new_polygon = np.stack(out, axis=1).astype('float32')
            resample_line.append(np.array(new_polygon))
        return resample_line

    def polygon_rotate(self, polygons):
        # try:
        for i, poly in enumerate(polygons):
            poly = poly[0].reshape(-1, 2)
            if poly.shape[0] == 4 and np.linalg.norm(poly[0] - poly[1], ord=2) * 1.5 < np.linalg.norm(
                    poly[0] - poly[-1], ord=2):
                poly = poly[[1, 2, 3, 0]]
            polygons[i] = [poly.reshape(-1)]
        # except Exception as e:
        #     print(e)
        return polygons

    def get_reference_points(self, polygons, length):
        def get_pp(line):
            ll = (line[1:] + line[:-1]) / 2
            pp = np.concatenate([line[:1], ll, line[-1:]])
            return pp

        n = polygons.shape[0]
        if n == length * 2:
            top_line, bot_line = polygons[:n // 2], polygons[n // 2:]
            center_line = (top_line + bot_line[::-1]) / 2
            reference_points = center_line
            ref_top, ref_bot = top_line, bot_line[::-1]
        else:
            top_line, bot_line = self.resample_polygon(polygons[:n // 2], polygons[n // 2:], length - 1)
            center_line = (top_line + bot_line[::-1]) / 2
            reference_points = get_pp(center_line)
            ref_top, ref_bot = get_pp(top_line), get_pp(bot_line[::-1])
        # l1 = ref_top - reference_points
        # l2 = reference_points[1:] - reference_points[:-1]
        # l2 = np.concatenate([l2, reference_points[-1:] - reference_points[-2:-1]])
        # dist = np.linalg.norm(l1, axis=-1)/np.linalg.norm([reference_points[0], reference_points[-1]])
        # cost = (l1 * l2).sum(-1)/(np.linalg.norm(l1) * np.linalg.norm(l2))
        # reference_points =

        # reference_points_info = np.concatenate([reference_points, dist[:,None], cost[:,None]], axis=1)
        reference_points_info = np.concatenate([reference_points, ref_top], axis=1)

        # return reference_points
        return reference_points_info

    def strQ2B(self, ustring):
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    def generate_targets(self, results):
        # if results['img_info']['file_name'] in ['train_ReCTS_000233.jpg','train_ReCTS_000246.jpg','train_ReCTS_000265.jpg','train_ReCTS_001034.jpg']:
        #     print(results)

        polygons = results['gt_masks'].masks
        polygons = self.polygon_rotate(polygons)
        polygon_masks_ignore = results['gt_masks_ignore'].masks
        gt_texts = results['texts']
        h, w, _ = results['img_shape']
        reference_points = []
        texts = []
        # print(gt_texts)
        for i, poly in enumerate(polygons):
            poly = poly[0].reshape(-1, 2)
            gt_texts[i] = gt_texts[i].lower()
            text_length = len(gt_texts[i])
            # if text_length == 0
            # try:
            if text_length == 0:
                continue
            try:
                ref_point = self.get_reference_points(poly, text_length + 2)
                reference_points.append(ref_point)
                texts.append(
                    self.strQ2B(gt_texts[i])
                )
            except Exception as e:
                print(e)
                continue

        # try:
        text_region = self.generate_text_region_mask((h, w), polygons)
        text_gauss_start, center_line_mask = self.generate_start_map((h, w), reference_points, polygons)
        # except Exception as e:
        #     print(e)
        #
        # text_region = (text_region + text_gauss_start) > 0.001
        # raise ValueError(0)
        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            # 'p3_maps': level_maps[0],
            # 'p4_maps': level_maps[1],
            # 'p5_maps': level_maps[2],
            # 'lv_text_polys_idx': lv_text_polys_idx,
            # 'polygons_area': polygons_area,
            # 'text_region': text_region,
            # 'text_start': text_gauss_start,
            'gt_texts': DC(texts, cpu_only=True),
            'gt_reference_points': reference_points,
            'gt_text_mask': text_region,
            'gt_head_mask': text_gauss_start,
            'gt_center_mask': center_line_mask,
            # 'gt_shape': results['img_info']['shape'],
            # 'gt_color': results['img_info']['color'],
            # 'lv_tps_coeffs': lv_tps_coeffs
        }
        for key, value in mapping.items():
            results[key] = value

        # img = results['img']
        # for points in reference_points:
        #     for point in points:
        #         img = cv2.circle(img, [round(point[0]), round(point[1])], 5, (255, 0, 0), 5)
        # img_path = os.path.join('gt_temp', results['img_info']['file_name'].split('/')[-1])
        # flag = cv2.imwrite(img_path, img)
        # print(flag)
        return results

    def generate_targets_infer(self, results):
        # if results['img_info']['file_name'] in ['train_ReCTS_000233.jpg','train_ReCTS_000246.jpg','train_ReCTS_000265.jpg','train_ReCTS_001034.jpg']:
        #     print(results)

        polygons = results['gt_masks'][0].masks
        polygons = self.polygon_rotate(polygons)
        polygon_masks_ignore = results['gt_masks_ignore'][0].masks
        gt_texts = results['texts']
        h, w, _ = results['img_shape']
        reference_points = []
        texts = []
        # print(gt_texts)
        for i, poly in enumerate(polygons):
            poly = poly[0].reshape(-1, 2)
            gt_texts[i] = gt_texts[i].lower()
            text_length = len(gt_texts[i])
            # if text_length == 0
            # try:
            if text_length == 0:
                continue
            try:
                ref_point = self.get_reference_points(poly, text_length + 2)
                reference_points.append(ref_point)
                texts.append(
                    self.strQ2B(gt_texts[i])
                )
            except Exception as e:
                print(e)
                continue

        # try:
        text_region = self.generate_text_region_mask((h, w), polygons)
        text_gauss_start, center_line_mask = self.generate_start_map((h, w), reference_points, polygons)
        # except Exception as e:
        #     print(e)
        #
        # text_region = (text_region + text_gauss_start) > 0.001
        # raise ValueError(0)
        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            # 'p3_maps': level_maps[0],
            # 'p4_maps': level_maps[1],
            # 'p5_maps': level_maps[2],
            # 'lv_text_polys_idx': lv_text_polys_idx,
            # 'polygons_area': polygons_area,
            # 'text_region': text_region,
            # 'text_start': text_gauss_start,
            'gt_texts': DC(texts, cpu_only=True),
            'gt_reference_points': reference_points,
            'gt_text_mask': text_region,
            'gt_head_mask': text_gauss_start,
            'gt_center_mask': center_line_mask,
            # 'gt_shape': results['img_info']['shape'],
            # 'gt_color': results['img_info']['color'],
            # 'lv_tps_coeffs': lv_tps_coeffs
        }
        for key, value in mapping.items():
            results[key] = value

        return results