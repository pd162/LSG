from .transforms import RandomCropPolyInstances
from mmdet.datasets.builder import PIPELINES
import numpy as np
import mmcv
from mmdet.core import BitmapMasks, PolygonMasks


@PIPELINES.register_module()
class RandomCropPolyInstancesWithText(RandomCropPolyInstances):

    def __call__(self, results):
        if len(results[self.instance_key].masks) < 1:
            return results
        if np.random.random_sample() < self.crop_ratio:
            res = results.copy()
            crop_box = self.sample_crop_box(results['img'].shape, results)
            results['crop_region'] = crop_box
            img = self.crop_img(results['img'], crop_box)
            results['img'] = img
            results['img_shape'] = img.shape

            # crop and filter masks
            x1, y1, x2, y2 = crop_box
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            labels = results['gt_labels']
            texts = results['texts']
            valid_labels = []
            valid_texts = []
            for key in results.get('mask_fields', []):
                if len(results[key].masks) == 0:
                    continue
                results[key] = results[key].crop(crop_box)
                # filter out polygons beyond crop box.
                masks = results[key].masks
                valid_masks_list = []

                for ind, mask in enumerate(masks):
                    assert len(mask) == 1
                    polygon = mask[0].reshape((-1, 2))
                    if (polygon[:, 0] >
                        -4).all() and (polygon[:, 0] < w + 4).all() and (
                            polygon[:, 1] > -4).all() and (polygon[:, 1] <
                                                           h + 4).all():
                        mask[0][::2] = np.clip(mask[0][::2], 0, w)
                        mask[0][1::2] = np.clip(mask[0][1::2], 0, h)
                        if key == self.instance_key:
                            valid_labels.append(labels[ind])
                            valid_texts.append(texts[ind])
                        valid_masks_list.append(mask)

                results[key] = PolygonMasks(valid_masks_list, h, w)
            results['gt_labels'] = np.array(valid_labels)
            results['texts'] = valid_texts
            if len(valid_labels)==0:
                print(results)

        return results


@PIPELINES.register_module()
class ResizePad:

    def __init__(self,
                 target_size,
                 pad_ratio=0.6,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0)):
        """Resize or pad images to be square shape.

        Args:
            target_size (int): The target size of square shaped image.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rescales image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        assert isinstance(target_size, int) or isinstance(target_size, tuple)
        assert isinstance(pad_ratio, float)
        assert isinstance(pad_with_fixed_color, bool)
        assert isinstance(pad_value, tuple)

        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def resize_img(self, img, keep_ratio=True):
        h, w, _ = img.shape
        if isinstance(self.target_size, int):
            if keep_ratio:
                t_h = self.target_size if h >= w else int(h * self.target_size / w)
                t_w = self.target_size if h <= w else int(w * self.target_size / h)
            else:
                t_h = t_w = self.target_size
        else:
            if keep_ratio:
                ratio = min(self.target_size[0]/h, self.target_size[1]/w)
                t_h = int(h*ratio)
                t_w = int(w*ratio)
            else:
                t_h = self.target_size[0]
                t_w = self.target_size[1]
        img = mmcv.imresize(img, (t_w, t_h))
        return img, (t_h, t_w)

    def pad(self, img):
        h, w = img.shape[:2]
        if h / w == self.target_size[0]/self.target_size[1]:
            return img, (0, 0)
        # pad_size = max(h, w)
        pad_h, pad_w = self.target_size
        if self.pad_with_fixed_color:
            expand_img = np.ones((pad_h, pad_w, 3), dtype=np.uint8)
            expand_img[:] = self.pad_value
        else:
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            expand_img = mmcv.imresize(img_cut, (pad_w, pad_h))
        # if h > w:
        #     y0, x0 = 0, (h - w) // 2
        # else:
        #     y0, x0 = (w - h) // 2, 0
        y0, x0 = (pad_h-h)//2, (pad_w-w)//2
        expand_img[y0:y0 + h, x0:x0 + w] = img
        offset = (x0, y0)

        return expand_img, offset

    def pad_mask(self, points, offset):
        x0, y0 = offset
        pad_points = points.copy()
        pad_points[::2] = pad_points[::2] + x0
        pad_points[1::2] = pad_points[1::2] + y0
        return pad_points

    def __call__(self, results):
        img = results['img']

        if np.random.random_sample() < self.pad_ratio:
            img, out_size = self.resize_img(img, keep_ratio=True)
            img, offset = self.pad(img)
        else:
            img, out_size = self.resize_img(img, keep_ratio=False)
            offset = (0, 0)

        results['img'] = img
        results['img_shape'] = img.shape

        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                continue
            results[key] = results[key].resize(out_size)
            masks = results[key].masks
            processed_masks = []
            for mask in masks:
                square_pad_mask = self.pad_mask(mask[0], offset)
                processed_masks.append([square_pad_mask])

            results[key] = PolygonMasks(processed_masks, *(img.shape[:2]))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
