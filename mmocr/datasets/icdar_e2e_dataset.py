from mmocr.datasets.icdar_dataset import IcdarDataset
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation import eval_hmean
import mmocr.utils as utils
from mmocr.core.evaluation import eval_seal
from mmocr.core.evaluation import eval_text

def id2text(rec):
    voc = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
     '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd',
     'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{',
     '|', '}', '~']
    text = ''
    for r in rec:
        if r == 96:
            break
        text += voc[r]
    return text


def _bezier_to_poly(bezier):
    # bezier to polygon
    bezier = np.array(bezier)
    u = np.linspace(0, 1, 8)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points.reshape(-1).tolist()



@DATASETS.register_module()
class IcdarE2EDataset(IcdarDataset):

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        if not self.test_mode:
            results['texts'] = results['ann_info']['texts']
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore, seg_map. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ignore = []
        gt_masks_ann = []

        gt_texts_ignore = []
        gt_texts_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            trans = ann.get('transcription', "")
            # ann['segmentation'] = _bezier_to_poly(ann.get('bezier_pts', None))
            # ann['transcription'] = id2text(trans)
            # if len(ann.get('segmentation', None)[0]) < 8:
            #     ann['segmentation'] = [[bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]]
            if ann.get('iscrowd', False) or trans == '###':
                gt_bboxes_ignore.append(bbox)
                gt_masks_ignore.append(ann.get(
                    'segmentation', None))  # to float32 for latter processing
                gt_texts_ignore.append(ann.get(
                    'rec', None
                ))
            #TODO Ignore according to transcription
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                trans = ann.get('transcription', None)
                if isinstance(trans, list) > 1:
                    trans = ''.join(trans)
                gt_texts_ann.append(trans)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        for i in range(len(gt_masks_ann)):
            gt_masks_ann[i][::2] = np.clip(gt_masks_ann[i][::2], 0, img_info['width'])
            gt_masks_ann[i][1::2] = np.clip(gt_masks_ann[i][1::2], 0, img_info['height'])


        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            texts=gt_texts_ann,
            texts_ignore=gt_texts_ignore
        )

        return ann

    def prepare_test_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results['texts'] = results['ann_info']['texts']
        results['masks'] = results['ann_info']['masks']
        return self.pipeline(results)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            try:
                if self.get_ann_info(idx)['bboxes'].shape[0] == 0:
                    idx = self._rand_another(idx)
                    continue
                data = self.prepare_train_img(idx)
                assert len(data['gt_texts'].data) != 0, "no texts in the image"
                assert data is not None
                assert len(data['gt_texts'].data) < 200, "too much texts in the image"
                return data
            except Exception as e:
                print(e)
                print('data error')
                idx = self._rand_another(idx)

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 logger=None,
                 score_thr=0.1,
                 rank_list=None,
                 lexicon_type=2,
                 with_lexicon=False,
                 **kwargs):

        """Evaluate the hmean metric.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[dict[str: float]]: The evaluation results.
        """
        # results = [{'boundary_result': r[0].tolist()} for r in results]
        # for r in results:
        #     bb = r['boundary_result']
        #     boundaries = []
        #     for b in bb:
        #         boundaries.append([b[0],b[1],b[2],b[1],b[2],b[3],b[0],b[3],b[4]])
        #     r['boundary_result'] = boundaries
        assert utils.is_type_list(results, dict)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-e2e', 'eval_seal', 'eval_text']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_info = {'filename': self.data_infos[i]['file_name']}
            img_infos.append(img_info)
            ann_infos.append(self.get_ann_info(i))
        # if 'totaltext' in self.ann_file:
        #     dataset_name = 'totaltext'
        # if 'ctw' in self.ann_file:
        #     dataset_name = 'ctw1500'
        if 'total' in self.ann_file.lower():
            gt_folder = 'evaluation/gt/gt_totaltext'
            IS_WORDSPOTTING = True
            lexicon_paths = ['', 'evaluation/lexicons/totaltext/weak_voc_new.txt', ]
            pair_paths = ['', 'evaluation/lexicons/totaltext/weak_voc_pair_list.txt', ]
            lexicon_type = 1
        elif 'ctw' in self.ann_file.lower():
            gt_folder = 'evaluation/gt/gt_ctw1500'
            IS_WORDSPOTTING = False
            lexicon_paths = ['', 'evaluation/lexicons/ctw1500/weak_voc_new.txt', ]
            pair_paths = ['', 'evaluation/lexicons/ctw1500/weak_voc_pair_list.txt', ]
            lexicon_type = 1
        elif 'ic13' in self.ann_file.lower():
            gt_folder = 'evaluation/gt/gt_ic13'
            IS_WORDSPOTTING = False
            lexicon_paths = [
                'evaluation/lexicons/ic13/GenericVocabulary_new.txt',
                'evaluation/lexicons/ic13/ch2_test_vocabulary_new.txt',
                'evaluation/lexicons/ic13/new_strong_lexicon/new_voc_img_',
            ]
            pair_paths = [
                'evaluation/lexicons/ic13/GenericVocabulary_pair_list.txt',
                'evaluation/lexicons/ic13/ch2_test_vocabulary_pair_list.txt',
                'evaluation/lexicons/ic13/new_strong_lexicon/pair_voc_img_',
            ]
            lexicon_type = lexicon_type
        elif 'ic15' in self.ann_file.lower():
            gt_folder = 'evaluation/gt/gt_ic15'
            IS_WORDSPOTTING = True
            lexicon_paths = [
                'evaluation/lexicons/ic15/GenericVocabulary_new.txt',
                'evaluation/lexicons/ic15/ch4_test_vocabulary_new.txt',
                'evaluation/lexicons/ic15/new_strong_lexicon/new_voc_img_',
            ]
            pair_paths = [
                'evaluation/lexicons/ic15/GenericVocabulary_pair_list.txt',
                'evaluation/lexicons/ic15/ch4_test_vocabulary_pair_list.txt',
                'evaluation/lexicons/ic15/new_strong_lexicon/pair_voc_img_',
            ]
            lexicon_type = lexicon_type
        elif 'inverse' in self.ann_file.lower():
            gt_folder = 'evaluation/gt/gt_inversetext'
            IS_WORDSPOTTING = True
            lexicon_paths = ['', 'evaluation/lexicons/inversetext/inversetext_lexicon.txt', ]
            pair_paths = ['', 'evaluation/lexicons/inversetext/inversetext_pair_list.txt', ]
            lexicon_type = 1
        else:
            raise ValueError('Cannot determine target dataset')

        if with_lexicon:
            lexicon_path = lexicon_paths[lexicon_type]
            pair_path = pair_paths[lexicon_type]
            lexicons = read_lexicon(lexicon_path)
            pairs = read_pair(pair_path)
        else:
            lexicons = None
            pairs = None

        if 'hmean-iou' in metrics:
            eval_results = eval_hmean(
                results,
                img_infos,
                ann_infos,
                metrics=metrics,
                score_thr=score_thr,
                logger=logger,
                rank_list=rank_list
            )

        if 'eval_seal' in metrics:
            # from mmocr.core.evaluation import eval_seal
            eval_results = eval_seal(results, self.coco, logger=logger)

        best_eval_results = dict()
        if 'eval_text' in metrics:
            new_results = []
            # post_process spts results
            for i, img_res in enumerate(results):
                for j, inst in enumerate(img_res['strs']):
                    char_len = len(inst)
                    try:
                        rec_score = np.array(img_res['char_scores'][j]).cumprod()[-1]
                        img_no = int(img_res['img_metas'][0]['filename'].split('/')[-1].split('.')[0])
                        if 'ic15' in self.ann_file.lower():
                            img_no = int(img_res['img_metas'][0]['filename'].split('/')[-1].split('.')[0][4:])
                        temp_dict = dict(
                            image_id=img_no,
                            polys=[img_res['reference_points'][j][(char_len / 2).__ceil__()]],
                            rec=inst,
                            # score=img_res['det_scores'][j] * rec_score
                            score=rec_score  # use_gt
                        )
                        new_results.append(temp_dict)
                    except:
                        print(np.array(img_res['char_scores'][j]))
                        continue
            submit_res = read_result(
                new_results,
                lexicons,
                pairs,
                0.3,
                gt_folder,
                lexicon_type
            )
            best_f = 0.

            for thr in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                eval_results = eval_text(submit_res, gt_folder, logger=logger, isspotting=IS_WORDSPOTTING, conf_thres=thr)
                if eval_results['F-Measure'] > best_f:
                    best_eval_results = eval_results
                    best_f = eval_results['F-Measure']
        return best_eval_results

import os
import json
import tqdm
import editdistance as ed

def read_lexicon(lexicon_path):
    if lexicon_path.endswith('.txt'):
        lexicon = open(lexicon_path, 'r').read().splitlines()
        lexicon = [ele.strip() for ele in lexicon]
    else:
        lexicon = {}
        lexicon_dir = os.path.dirname(lexicon_path)
        num_file = len(os.listdir(lexicon_dir))
        assert (num_file % 2 == 0)
        for i in range(num_file // 2):
            lexicon_path_ = lexicon_path + f'{i + 1:d}.txt'
            lexicon[i] = read_lexicon(lexicon_path_)
    return lexicon


def read_pair(pair_path):
    if 'ctw1500' in pair_path:
        return None

    if pair_path.endswith('.txt'):
        pair_lines = open(pair_path, 'r').read().splitlines()
        pair = {}
        for line in pair_lines:
            line = line.strip()
            word = line.split(' ')[0].upper()
            word_gt = line[len(word) + 1:]
            pair[word] = word_gt
    else:
        pair = {}
        pair_dir = os.path.dirname(pair_path)
        num_file = len(os.listdir(pair_dir))
        assert (num_file % 2 == 0)
        for i in range(num_file // 2):
            pair_path_ = pair_path + f'{i + 1:d}.txt'
            pair[i] = read_pair(pair_path_)
    return pair


def read_result(results, lexicons, pairs, match_dist_thres, gt_folder, lexicon_type):
    results.sort(reverse=True, key=lambda x: x['score'])

    results = [result for result in results if len(result['rec']) > 0]

    if not lexicons is None:
        print('Processing Results using Lexicon')
        new_results = []
        for result in tqdm.tqdm(results):
            rec = result['rec']
            if lexicon_type == 2:
                lexicon = lexicons[result['image_id'] - 1]
                pair = pairs[result['image_id'] - 1]
            else:
                lexicon = lexicons
                pair = pairs

            match_word, match_dist = find_match_word(rec, lexicon, pair)
            if match_dist < match_dist_thres or \
               (('gt_ic13' in gt_folder or 'gt_ic15' in gt_folder) and lexicon_type == 0):
                rec = match_word
            else:
                continue
            result['rec'] = rec
            new_results.append(result)
        results = new_results

    return results


def find_match_word(rec_str, lexicon, pair):
    rec_str = rec_str.upper()
    match_word = ''
    match_dist = 100
    for word in lexicon:
        word = word.upper()
        ed_dist = ed.eval(rec_str, word)
        norm_ed_dist = ed_dist / max(len(word), len(rec_str))
        if norm_ed_dist < match_dist:
            match_dist = norm_ed_dist
            if pair:
                match_word = pair[word]
            else:
                match_word = word
    return match_word, match_dist

