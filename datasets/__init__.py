from pathlib import Path
import torch
import torch.utils.data
from .torchvision_datasets import CocoDetection

from .dataset import build_train
from .val_support_dataset import build_val_support_dataset


# Meta-settings for few-shot object detection: base / novel category split
coco_base_class_ids = [
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

coco_novel_class_ids = [
    1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72
]

voc_base1_class_ids = [
    1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20
]

voc_novel1_class_ids = [
    3, 6, 10, 14, 18
]

voc_base2_class_ids = [
    2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20
]

voc_novel2_class_ids = [
    1, 5, 10, 13, 18
]

voc_base3_class_ids = [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 19, 20
]

voc_novel3_class_ids = [
    4, 8, 14, 17, 18
]


def get_class_ids(dataset, type):
    if dataset == 'coco':
        if type == 'all':
            ids = (coco_base_class_ids + coco_novel_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return coco_base_class_ids
        elif type == 'novel':
            return coco_novel_class_ids
        else:
            raise ValueError
   
    if dataset == 'voc1':
        if type == 'all':
            ids = (voc_base1_class_ids + voc_novel1_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return voc_base1_class_ids
        elif type == 'novel':
            return voc_novel1_class_ids
        else:
            raise ValueError
    if dataset == 'voc2':
        if type == 'all':
            ids = (voc_base2_class_ids + voc_novel2_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return voc_base2_class_ids
        elif type == 'novel':
            return voc_novel2_class_ids
        else:
            raise ValueError
    if dataset == 'voc3':
        if type == 'all':
            ids = (voc_base3_class_ids + voc_novel3_class_ids)
            ids.sort()
            return ids
        elif type == 'base':
            return voc_base3_class_ids
        elif type == 'novel':
            return voc_novel3_class_ids
        else:
            raise ValueError
   
    raise ValueError


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args): 
    assert image_set in ['base_train', 'fewshot_finetune', 'val', 'val_support_dataset'], "image_set must be 'base_train', 'fewshot_finetune', 'val' or 'val_support_dataset'."
    assert args.dataset_file in ['coco', 'voc1', 'voc2', 'voc3'], "dataset_file must be 'coco', 'voc1', 'voc2' or 'voc3'."
    # For training set, need to perform base/novel category filtering.
    # For training set, we use dataset with support to construct meta-tasks
    if image_set == 'base_train': 
        assert args.fewshot_finetune == False
        if args.dataset_file == 'coco':
            root = Path('data/coco')
            img_folder = root / "train2017"
            ann_file = root / "annotations" / 'instances_train2017.json'
            class_ids = coco_base_class_ids
        if args.dataset_file in ['voc1', 'voc2', 'voc3']:
            root = Path('data/voc')
            img_folder = root / "images"
            ann_file = root / "annotations" / 'pascal_trainval0712.json'
            if args.dataset_file == 'voc1':
                class_ids = voc_base1_class_ids
            if args.dataset_file == 'voc2':
                class_ids = voc_base2_class_ids
            if args.dataset_file == 'voc3':
                class_ids = voc_base3_class_ids
         
        print('\nbase training')
        print(f'img_folder: {img_folder}')
        print(f'ann_file: {ann_file}')
        print(f"base training class_ids: {class_ids}")
        return build_train(args, img_folder, ann_file, image_set, activated_class_ids=class_ids, with_support=True)
        
        

    
    if image_set == 'fewshot_finetune':
        assert args.fewshot_finetune == True

        if args.dataset_file == 'coco':
            root = Path('data/coco')
            img_folder = root / "train2017"
            ann_file = root.parent / 'coco_fewshot' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
            class_ids = coco_base_class_ids + coco_novel_class_ids
            class_ids.sort()
        if args.dataset_file in ['voc1', 'voc2', 'voc3']:
            root = Path('data/voc')
            img_folder = root / "images"
            if args.dataset_file == 'voc1':
                ann_file = root.parent / 'voc_fewshot_split1' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
            if args.dataset_file == 'voc2':
                ann_file = root.parent / 'voc_fewshot_split2' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
            if args.dataset_file == 'voc3':
                ann_file = root.parent / 'voc_fewshot_split3' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json' 

            class_ids = list(range(1, 20+1))
        
        print('\nfew-shot:')
        print(f'img_folder: {img_folder}')
        print(f'ann_file: {ann_file}')
        print(f"few-shot finetune class_ids: {class_ids}")
        return build_train(args, img_folder, ann_file, image_set, activated_class_ids=class_ids, with_support=True)
        



    # For valid set, no need to perform base/novel category filtering.
    # This is because that evaluation should be performed on all images.
    # For valid set, support dataset should be constructed from the base_train / fewshot_finetune dataset.

    if image_set == 'val': 
        if args.dataset_file == 'coco':
            root = Path('data/coco')
            img_folder = root / "val2017"
            ann_file = root / "annotations" / 'instances_val2017.json'
            class_ids = coco_base_class_ids + coco_novel_class_ids 
            class_ids.sort()

        if args.dataset_file in ['voc1', 'voc2', 'voc3']:
            root = Path('data/voc')
            img_folder = root / "images"
            ann_file = root / "annotations" / 'pascal_test2007.json'
            class_ids = list(range(1, 20+1))

        print('\nval:')
        print(f'img_folder: {img_folder}')
        print(f'ann_file: {ann_file}')
        print(f"val class_ids: {class_ids}")
        return build_train(args, img_folder, ann_file, image_set, activated_class_ids=class_ids, with_support=False)



    if image_set == 'val_support_dataset': 
        if args.dataset_file == 'coco':
            root = Path('data/coco')
            img_folder = root / "train2017"
            if not args.fewshot_finetune: 
                ann_file = root / "annotations" / 'instances_train2017.json'
                class_ids = coco_base_class_ids
            else:
                ann_file = root.parent / 'coco_fewshot' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
                class_ids = coco_base_class_ids + coco_novel_class_ids
                class_ids.sort()
                


        if args.dataset_file in ['voc1', 'voc2', 'voc3']:
            root = Path('data/voc')
            img_folder = root / "images"
            if not args.fewshot_finetune: 
                ann_file = root / "annotations" / 'pascal_trainval0712.json'

                if args.dataset_file == 'voc1':
                    class_ids = voc_base1_class_ids
                if args.dataset_file == 'voc2':
                    class_ids = voc_base2_class_ids
                if args.dataset_file == 'voc3':
                    class_ids = voc_base3_class_ids
                
            else: 
                if args.dataset_file == 'voc1':
                    ann_file = root.parent / 'voc_fewshot_split1' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
                if args.dataset_file == 'voc2':
                    ann_file = root.parent / 'voc_fewshot_split2' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
                if args.dataset_file == 'voc3':
                    ann_file = root.parent / 'voc_fewshot_split3' / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'

                class_ids = list(range(1, 20+1))

        print('\nval_support:')
        print(f'img_folder: {img_folder}')
        print(f'ann_file: {ann_file}')
        print(f"val_support class_ids: {class_ids}")
        return build_val_support_dataset(args, img_folder, ann_file, class_ids)