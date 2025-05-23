#!/usr/bin/env python
import os.path as osp
from argparse import ArgumentParser

import mmcv
from mmcv.utils import ProgressBar
from mmocr.apis import init_detector, model_inference
from mmocr.models import build_detector  # noqa: F401
from mmocr.utils import list_from_file, list_to_file


def gen_target_path(target_root_path, src_name, suffix):
    """Gen target file path.

    Args:
        target_root_path (str): The target root path.
        src_name (str): The source file name.
        suffix (str): The suffix of target file.
    """
    assert isinstance(target_root_path, str)
    assert isinstance(src_name, str)
    assert isinstance(suffix, str)

    file_name = osp.split(src_name)[-1]
    name = osp.splitext(file_name)[0]
    return osp.join(target_root_path, name + suffix)


def save_results(result, out_dir, img_name, score_thr=0.3):
    """Save result of detected bounding boxes (quadrangle or polygon) to txt
    file.

    Args:
        result (dict): Text Detection result for one image.
        img_name (str): Image file name.
        out_dir (str): Dir of txt files to save detected results.
        score_thr (float, optional): Score threshold to filter bboxes.
    """
    assert 'strs' in result
    assert score_thr > 0 and score_thr < 1

    txt_file = gen_target_path(out_dir, img_name, '.txt')
    valid_rec_res = [
        res for i,res in enumerate(result['strs']) if result['det_scores'][i] > score_thr
    ]
    lines = valid_rec_res
    list_to_file(txt_file, lines)


def main():
    parser = ArgumentParser()
    parser.add_argument('img_root', type=str, help='Image root path')
    # parser.add_argument('img_list', type=str, help='Image path list file')
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='Bbox score threshold')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./results',
        help='Dir to save '
             'results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    args = parser.parse_args()

    assert 0 < args.score_thr < 1

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if hasattr(model, 'module'):
        model = model.module

    # Start Inference
    out_vis_dir = osp.join(args.out_dir, 'out_vis_dir')
    #mmcv.mkdir_or_exist(out_vis_dir)
    out_txt_dir = osp.join(args.out_dir, 'out_txt_dir')
    mmcv.mkdir_or_exist(out_txt_dir)

    # lines = list_from_file(args.img_list)
    import os
    lines = os.listdir(args.img_root)
    progressbar = ProgressBar(task_num=len(lines))
    for line in lines:
        progressbar.update()
        try:
            progressbar.update()
            img_path = osp.join(args.img_root, line.strip())
            if not osp.exists(img_path):
                raise FileNotFoundError(img_path)
            # Test a single image
            result = model_inference(model, img_path)
            img_name = osp.basename(img_path)
            # # save result
            save_results(result, out_txt_dir, img_name, score_thr=args.score_thr)
        except Exception as e:
            print(e)
            continue
        # out_vis_dir = args.out_dir
        out_file = osp.join(out_vis_dir, img_name)
        kwargs_dict = {
            'score_thr': args.score_thr,
            'show': False,
            'out_file': out_file
        }
        model.show_result(img_path, result, **kwargs_dict)

    print(f'\nInference done, and results saved in {args.out_dir}\n')


if __name__ == '__main__':
    main()

