import os
import shutil
from pathlib import Path
import json

def combine_coco_annotations(annotations_list):
    """
    Объединяет аннотации в формате COCO.
    
    Args:
        annotations_list (list): Список словарей с аннотациями COCO
        
    Returns:
        dict: Объединенные аннотации в формате COCO
    """
    print(f"Получено аннотаций для объединения: {len(annotations_list)}")
    
    if not annotations_list:
        print("Список аннотаций пуст!")
        return None
        
    # Создаем базовую структуру COCO
    combined = {
        "info": annotations_list[0].get("info", {}),
        "licenses": annotations_list[0].get("licenses", []),
        "categories": annotations_list[0].get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # Счетчики для уникальных ID
    image_id_counter = 1
    annotation_id_counter = 1
    
    # Словарь для отслеживания соответствия старых и новых ID изображений
    image_id_mapping = {}
    
    # Объединяем изображения и аннотации
    for ann in annotations_list:
        # Обрабатываем изображения
        for img in ann.get("images", []):
            old_id = img["id"]
            new_id = image_id_counter
            image_id_mapping[old_id] = new_id
            
            # Создаем новую запись изображения
            new_img = img.copy()
            new_img["id"] = new_id
            combined["images"].append(new_img)
            
            image_id_counter += 1
        
        # Обрабатываем аннотации
        for ann_item in ann.get("annotations", []):
            # Создаем новую запись аннотации
            new_ann = ann_item.copy()
            new_ann["id"] = annotation_id_counter
            new_ann["image_id"] = image_id_mapping[ann_item["image_id"]]
            combined["annotations"].append(new_ann)
            
            annotation_id_counter += 1
    
    print(f"Объединено изображений: {len(combined['images'])}")
    print(f"Объединено аннотаций: {len(combined['annotations'])}")
    return combined

def combine_dataset(source_dir, target_dir, preserve_names=True):
    """
    Объединяет изображения и JSON-файлы в формате COCO из подпапок в один датасет.
    
    Args:
        source_dir (str): Путь к исходной папке с подпапками
        target_dir (str): Путь к целевой папке для объединенного датасета
        preserve_names (bool): Сохранять ли оригинальные имена файлов
    """
    try:
        # Создаем целевую директорию, если она не существует
        os.makedirs(target_dir, exist_ok=True)
        
        # Словарь для хранения информации о файлах
        files_info = {}
        # Список для хранения всех аннотаций COCO
        coco_annotations = []
        
        # Получаем список всех подпапок
        subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]
        print(f"Найдено подпапок: {len(subfolders)}")
        
        # Счетчик для уникальных имен файлов
        file_counter = 0
        
        # Проходим по каждой подпапке
        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder)
            print(f"\nОбработка папки: {folder_name}")
            
            # Проверяем наличие папки annotations в текущей подпапке
            annotations_path = os.path.join(subfolder, 'annotations')
            if os.path.exists(annotations_path) and os.path.isdir(annotations_path):
                print(f"Найдена папка annotations в {folder_name}")
                # Читаем все JSON файлы в папке annotations
                for json_file in os.listdir(annotations_path):
                    if json_file.endswith('.json'):
                        json_path = os.path.join(annotations_path, json_file)
                        print(f"Обработка файла аннотаций: {json_file}")
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                annotations = json.load(f)
                                # Добавляем аннотации в список
                                coco_annotations.append(annotations)
                                print(f"Успешно загружены аннотации из {json_file}")
                        except Exception as e:
                            print(f"Ошибка при чтении файла {json_path}: {str(e)}")
                
            # Получаем список всех файлов в подпапке
            for filename in os.listdir(subfolder):
                # Проверяем, является ли файл изображением
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Формируем пути к исходному и целевому файлам
                    source_path = os.path.join(subfolder, filename)
                    
                    if preserve_names:
                        # Гарантируем уникальность имени файла, добавляя имя папки
                        base_name, ext = os.path.splitext(filename)
                        target_filename = f"{folder_name}_{base_name}{ext}"
                    else:
                        # Генерируем новое имя файла
                        target_filename = f"image_{file_counter}{Path(filename).suffix}"
                    
                    target_path = os.path.join(target_dir, target_filename)
                    
                    # Копируем файл
                    shutil.copy2(source_path, target_path)
                    print(f"Скопирован файл: {source_path} -> {target_path}")
                    
                    # Сохраняем информацию о файле
                    files_info[target_filename] = {
                        'original_path': source_path,
                        'source_folder': folder_name,
                        'original_name': filename
                    }
                    
                    file_counter += 1
        
        # Сохраняем информацию о файлах в JSON
        info_file = os.path.join(target_dir, 'files_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(files_info, f, ensure_ascii=False, indent=4)
        
        # Объединяем аннотации COCO
        print(f"\nНайдено файлов аннотаций: {len(coco_annotations)}")
        if coco_annotations:
            print("Начинаем объединение аннотаций...")
            combined_annotations = combine_coco_annotations(coco_annotations)
            if combined_annotations:
                annotations_file = os.path.join(target_dir, 'combined_annotations.json')
                with open(annotations_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_annotations, f, ensure_ascii=False, indent=4)
                print(f"Объединенные аннотации COCO сохранены в {annotations_file}")
            else:
                print("Ошибка: не удалось создать объединенные аннотации")
        else:
            print("Предупреждение: не найдено файлов аннотаций для объединения")
        
        print(f"\nОбъединение завершено! Всего скопировано {file_counter} изображений.")
        print(f"Информация о файлах сохранена в {info_file}")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    # Пути к папкам
    source_directory = "Dataset"  # Исходная папка с подпапками
    target_directory = "Combined_Dataset"  # Папка для объединенного датасета
    
    # Запускаем объединение
    combine_dataset(source_directory, target_directory, preserve_names=True)