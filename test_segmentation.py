import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os

def setup_cfg():
    cfg = get_cfg()
    #cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "output_maskrcnn_r101/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # порог уверенности для предсказаний
    return cfg

def test_segmentation(image_path, output_path):
    # Настройка конфигурации и создание предсказателя
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Получение предсказаний
    outputs = predictor(image)
    
    # Визуализация результатов
    v = Visualizer(image[:, :, ::-1], 
                  metadata=MetadataCatalog.get("territory_train"),
                  scale=0.8)
    
    # Отображение масок и bounding boxes
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Сохранение результата
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    
    # Вывод информации о предсказаниях
    instances = outputs["instances"].to("cpu")
    print(f"Найдено {len(instances)} объектов")
    print(f"Уверенность предсказаний: {instances.scores.tolist()}")
    
    return instances

if __name__ == "__main__":
    # Создаем директорию для результатов, если её нет
    os.makedirs("test_results", exist_ok=True)
    
    # Тестируем на изображении из валидационного набора
    test_image = "Combined_Dataset/images/Tenzin (2)_16.png"
    output_path = "test_results/result.jpg"
    
    instances = test_segmentation(test_image, output_path)
    print(f"Результаты сохранены в {output_path}") 
