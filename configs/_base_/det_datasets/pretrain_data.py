dataset_type = 'IcdarE2EDataset'

mlt19_train = dict(
    type=dataset_type,
    ann_file='xxx/proj/data/std/coco_format/icdar2019_mlt_en_train.json',
    img_prefix='xxx/proj/data/benchmark_datasets/mlt19/train_image',
    pipeline=None)

icdar13_train = dict(
    type=dataset_type,
    ann_file='xxx/proj/code/mmocr/mmocr_ic13_train.json',
    img_prefix='xxx/proj/data/std/icdar2013/train_images',
    pipeline=None)

icdar15_train = dict(
    type=dataset_type,
    ann_file='xxx/proj/data/std/icdar2015/icdar15_train.json',
    img_prefix='xxx/proj/data/std/icdar2015/ch4_training_images',
    pipeline=None)

icdar15_test = dict(
    type=dataset_type,
    ann_file='xxx/proj/data/std/icdar2015/icdar15_test.json',
    img_prefix='xxx/proj/data/std/icdar2015/ch4_test_images',
    pipeline=None)  # bug!

mlt17_train = dict(
    type=dataset_type,
    ann_file='xxx/proj/code/mmocr/mmocr_mlt17_train.json',
    img_prefix='xxx/proj/data/std/mlt2017/MLT_train_images',
    pipeline=None
)

textocr_train_1 = dict(
type=dataset_type,
    ann_file='xxx/proj/code/mmocr/mmocr_train_37voc_1.json',
    img_prefix='xxx/proj/data/std/textocr/train_images',
    pipeline=None
)

textocr_train_2 = dict(
type=dataset_type,
    ann_file='xxx/proj/code/mmocr/mmocr_train_37voc_2.json',
    img_prefix='xxx/proj/data/std/textocr/train_images',
    pipeline=None
)


tt_train = dict(
    type=dataset_type,
    ann_file='xxx/proj/code/mmocr/totaltext/mmocr_train.json',
    img_prefix='xxx/proj/code/mmocr/totaltext/train_images',
    pipeline=None
)

tt_test = dict(
    type=dataset_type,
    ann_file='xxx/proj/code/mmocr/totaltext/mmocr_test.json',
    img_prefix='xxx/proj/code/mmocr/totaltext/test_images',
    pipeline=None
)

my_synth_1 = dict(
    type=dataset_type,
    ann_file='xxx/proj/data/std/my_synthtext_v3/results_curved_straightv3_64k.json',
    img_prefix='xxx/proj/data/std/my_synthtext_v3/',
    pipeline=None
)

my_synth_2 = dict(
    type=dataset_type,
    ann_file='xxx/proj/data/std/my_synthtext_v3/results_curved_cosv3_64k.json',
    img_prefix='xxx/proj/data/std/my_synthtext_v3/',
    pipeline=None
)

my_synth_3 = dict(
    type=dataset_type,
    ann_file='xxx/proj/data/std/my_synthtext_v3/results_curved_circlev3_64k.json',
    img_prefix='xxx/proj/data/std/my_synthtext_v3/',
    pipeline=None
)

# mlt19 = dict(
#     type=dataset_type,
#     ann_file='/data1/ljh/data/mlt19/mlt19_train_English.json',
#     img_prefix='/data1/ljh/data/mlt19/allimgs',
#     pipeline=None
# )

ctw_train = dict(
    type=dataset_type,
    ann_file='xxx/proj/code/mmocr/mmocr_train_ctw1500_maxlen25_v2.json',
    img_prefix='xxx/proj/data/std/CTW1500/ctwtrain_text_image',
    pipeline=None

)
# synth ic13 ic15 tt ctw mlt17

train_list = [

    textocr_train_1,
    textocr_train_2,
    my_synth_1,
    my_synth_2,
    my_synth_3,
    mlt19_train,
    icdar13_train,
    icdar15_train,
    mlt17_train,
    tt_train,
    ctw_train,
]

test_list = [tt_test]
