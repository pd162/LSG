import sys
from mmcv.utils import print_log
import Levenshtein
import numpy as np


def eval_seal(results, coco_annos, logger=None):
    img_ids = coco_annos.get_img_ids()
    n_correct_rec = 0
    n_rec = 1e-8
    n_gt = 1e-8
    sum_dis = 0

    n_correct_shape = 0
    n_correct_color = 0
    for i, img_id in enumerate(img_ids):
        img_name = coco_annos.load_imgs([img_id])[0]['file_name']
        # idx = str(img_id).zfill(7)
        res = results[i]['strs']
        char_scores = results[i]['char_scores']
        det_scores = results[i]['det_scores']

        keep_pts = []
        keep_strs = []
        keep_char_scores = []
        keep_det_scores = []
        for di, ds in enumerate(det_scores):
            if ds > 0.05:
                keep_strs.append(res[di])
                keep_det_scores.append(ds)
                keep_char_scores.append(char_scores[di])
                keep_pts.append(results[i]['reference_points'][di][0])
        keep_pts = np.linalg.norm(np.array(keep_pts), axis=-1)
        sort_id = keep_pts.argsort()

        res = keep_strs
        char_scores = keep_char_scores
        det_scores = keep_det_scores

        ann_id = coco_annos.getAnnIds(img_id)
        gt_trans_lists = [an['transcription'].lower() for an in coco_annos.loadAnns(ann_id)]
        # gt_shape = [an['shape'] for an in coco_annos.loadImgs(img_id)]
        # gt_color = [an['color'] for an in coco_annos.loadImgs(img_id)]
        # gt_trans = ''.join(gt_trans)

        if len(res) == 0:
            res_trans_lists = []
        else:
            res_trans_lists = [res[idd].lower() for idd in sort_id]
        n_rec += len(res_trans_lists)
        # n_matched = 0

        # if gt_shape[0] == results[0]['shape'].item():
        #     n_correct_shape += 1
        # else:
        #     print("shape error!")
        # if gt_color[0] == results[0]['color'].item():
        #     n_correct_color += 1
        # else:
        #     print("color error!")
        for g in gt_trans_lists:
            if len(res_trans_lists) > 0:
                g_pres_dis = [Levenshtein.distance(g, pre) for pre in res_trans_lists]
                closest_idx = np.argmin(g_pres_dis)
                pre = res_trans_lists[closest_idx]
                min_dis = g_pres_dis[closest_idx]
                res_trans_lists.pop(closest_idx)
            else:
                pre = ""
                min_dis = Levenshtein.distance(g, pre)
            if min_dis == 0:
                n_correct_rec += 1
            #                print(g)
            ned = min_dis / (max(len(g), len(pre)) + 1e-8)
            sum_dis += ned

            n_gt += 1

    p = n_correct_rec / n_rec
    r = n_correct_rec / n_gt
    f = p * r * 2 / (p + r + 1e-8)
    # color_acc = n_correct_color / len(img_ids)
    # shape_acc = n_correct_shape / len(img_ids)
    met = {
        "1-N.E.D": 1 - sum_dis / n_gt,
        "Precision": p,
        "Recall": r,
        "F-Measure": f,
        # "color_acc": color_acc,
        # "shape_acc": shape_acc
    }
    print_log(met, logger)
    return met
