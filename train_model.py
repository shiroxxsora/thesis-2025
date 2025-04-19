import os
import json
import random
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import CfgNode as CN
import albumentations as A
from detectron2.data import detection_utils as utils
import torch
from detectron2.data import transforms as T
import copy
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util

class F2Evaluator(COCOEvaluator):
    def __init__(self, dataset_name, output_dir=None):
        super().__init__(dataset_name, output_dir=output_dir)
        self._f2_scores = []
        
    def _evaluate_predictions_on_coco(self, coco_gt, coco_results, iou_type="segm"):
        """
        Оценивает предсказания с использованием F2-score
        """
        results = super()._evaluate_predictions_on_coco(coco_gt, coco_results, iou_type)
        
        # Вычисляем F2-score
        if len(coco_results) > 0:
            f2_scores = []
            for result in coco_results:
                if "segmentation" in result:
                    pred_mask = mask_util.decode(result["segmentation"])
                    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[result["image_id"]]))
                    
                    if len(gt_anns) > 0:
                        gt_mask = mask_util.decode(gt_anns[0]["segmentation"])
                        
                        # Вычисляем TP, FP, FN
                        tp = np.sum(np.logical_and(pred_mask, gt_mask))
                        fp = np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
                        fn = np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))
                        
                        # Вычисляем precision и recall
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        
                        # Вычисляем F2-score
                        beta = 2  # Больший вес для recall
                        f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
                        f2_scores.append(f2)
            
            if f2_scores:
                avg_f2 = np.mean(f2_scores)
                results["segm"]["f2"] = avg_f2
                self._f2_scores.append(avg_f2)
                print(f"F2-score: {avg_f2:.4f}")
        
        return results

class CustomTrainer(DefaultTrainer):
    """
    Кастомный тренер с аугментацией данных
    """
    @classmethod
    def build_train_augmentation(cls):
        """
        Создает пайплайн аугментации для обучения
        """
        augmentation = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"
            ),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomRotation([-15, 15]),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomCrop("relative_range", (0.8, 0.8)),
            T.RandomLighting(0.7),
        ]
        return augmentation
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Создает эвалуатор с F2-score
        """
        return F2Evaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

def setup_cfg():
    """
    Создает конфигурацию для Mask R-CNN
    """
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("territory_train",)
    cfg.DATASETS.TEST = ("territory_val",)
    
    # Параметры обучения
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 7500
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.STEPS = [3000, 4500, 5500]
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # только один класс - территория
    
    # Параметры для изображений разного размера
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Настройка оценки
    cfg.TEST.EVAL_PERIOD = 500  # Оцениваем каждые 500 итераций
    
    return cfg

def split_dataset(annotations_file, train_ratio=0.8):
    """
    Разделяет датасет на тренировочную и валидационную части
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Получаем все image_ids
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)
    
    # Разделяем на train и val
    split_idx = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])
    
    # Создаем train и val аннотации
    train_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': [],
        'annotations': []
    }
    
    val_data = copy.deepcopy(train_data)
    
    # Распределяем изображения
    for img in data['images']:
        if img['id'] in train_ids:
            train_data['images'].append(img)
        else:
            val_data['images'].append(img)
    
    # Распределяем аннотации
    for ann in data['annotations']:
        if ann['image_id'] in train_ids:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)
    
    return train_data, val_data

def main():
    # Создаем директории для разделенного датасета
    os.makedirs('train_annotations', exist_ok=True)
    os.makedirs('val_annotations', exist_ok=True)
    
    # Разделяем датасет
    train_data, val_data = split_dataset('Combined_Dataset/combined_annotations.json')
    
    # Сохраняем разделенные аннотации
    with open('train_annotations/instances_train.json', 'w') as f:
        json.dump(train_data, f)
    with open('val_annotations/instances_val.json', 'w') as f:
        json.dump(val_data, f)
    
    # Регистрируем наборы данных
    register_coco_instances(
        "territory_train",
        {},
        "train_annotations/instances_train.json",
        "Combined_Dataset"
    )
    register_coco_instances(
        "territory_val",
        {},
        "val_annotations/instances_val.json",
        "Combined_Dataset"
    )
    
    # Настраиваем конфигурацию
    cfg = setup_cfg()
    
    # Создаем директорию для сохранения модели
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Создаем тренер и начинаем обучение
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main() 
