dataset_type = 'IcdarE2EDataset'

icdar15_train = dict(
    type=dataset_type,
    ann_file='/data1/xxx/data/ICDAR15/mmocr_ic15_train.json',
    img_prefix='/data1/xxx/code/mmocrdev/icdar2015/train_images',
    pipeline=None)

icdar15_test = dict(
    type=dataset_type,
    ann_file='/data1/xxx/code/mmocrdev/mmocr_ic15_test.json',
    img_prefix='/data1/xxx/code/mmocrdev/icdar2015/test_images',
    pipeline=None
)

tt_train = dict(
    type=dataset_type,
    ann_file='totaltext/mmocr_train.json',
    select_first_k=10,
    img_prefix='totaltext/train_images',
    pipeline=None
)

tt_test = dict(
    type=dataset_type,
    # ann_file='xxx/proj/code/mmocr/totaltext/mmocr_test.json',
    # img_prefix='/data1/xxx/code/mmocrdev/totaltext/test_images',
    # img_prefix='xxx/proj/code/mmocr/totaltext/test_images',
    ann_file='totaltext/mmocr_train.json',
    select_first_k=10,
    img_prefix='totaltext/train_images',
    pipeline=None
)

ctw_train = dict(
    type=dataset_type,
    ann_file='/data1/xxx/data/ctw1500/new_ctw1500_train.json',
    img_prefix='/data1/xxx/data/ctw1500/train/train_images_rotate',
    pipeline=None
)

ctw_test = dict(
    type=dataset_type,
    ann_file='/data1/xxx/data/ctw1500/new_ctw1500_test.json',
    img_prefix='/data1/xxx/data/ctw1500/test/test_images',
    pipeline=None
)

ivt_train = dict(

)

ivt_test = dict(
    type=dataset_type,
    ann_file='datasets/inversetext/test_poly.json',
    img_prefix='inversetext/test_images',
    pipeline=None
)

tt_test_small = dict(
    type=dataset_type,
    ann_file='xxx/proj/code/mmocr/totaltext_subset1.json',
    img_prefix='xxx/proj/code/mmocr/totaltext/test_images',
    pipeline=None
)

tt_test_normal = dict(
    type=dataset_type,
    ann_file='xxx/proj/code/mmocr/totaltext_subset2.json',
    img_prefix='xxx/proj/code/mmocr/totaltext/test_images',
    pipeline=None
)


tt_train_list = [tt_train]
tt_test_list = [tt_test]
icdar15_train_list = [icdar15_train]
icdar15_test_list = [icdar15_test]
ctw_train_list = [ctw_train]
ctw_test_list = [ctw_test]
ivt_test_list = [ivt_test]


finetune_train_list = [tt_train]
finetune_test_list = [tt_test]