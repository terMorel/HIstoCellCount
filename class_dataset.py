from torch.utils.data import Dataset
from os import listdir
from pathlib import Path
import cv2
import torch
import numpy as np

#DATASET_ROOT = "/content/drive/MyDrive/Colab Notebooks/HistoCellCount/dataset_small"
DATASET_ROOT = "/home/terenkovkirill/HistoCellCount/dataset_small"
NUM_CLASSES = 3
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class CellCountDataset(Dataset):
    def __init__(self, root_path, num_classes, transform = None):
        self.transform   =  transform
        self.num_classes = num_classes

        if root_path is not None:
            self.root_path = Path(root_path)
            self.path_to_images = self.root_path / "images"
            self.path_to_labels = self.root_path / "labels"
        else:
            raise ValueError("Не передан путь к корневой папке.")
        
        ids = self._collect_ids()
        self.samples = []

        for filename in ids:
            try:
                with open (self.path_to_labels / f"{filename}.txt") as file:
                    self.samples.append((filename, self._points_to_target(points = self._read_points(filename))))
                        
            except FileNotFoundError:
                print(f"Файл {filename}.txt не найден")
            except Exception as e:
                print(f"Произошла ошибка: {e}")
            
        self.len_dataset = len(self.samples)


    def __len__(self):
        return self.len_dataset
    
    
    def __getitem__(self, index):
        if index < 0 or index >= len(self.samples):
            raise IndexError("Индекс вне диапазона.")

        sample_id, target_list = self.samples[index]

        image_path = self.path_to_images / f"{sample_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if torch.is_tensor(image):
            image = image.float()                       #предполагаем, что данные нормализованны
        else:
            image = self._image_to_tensor(image)

        target = torch.tensor(target_list, dtype = torch.float32)

        return {
            "image":     image,
            "target":    target,
            "image_id":  sample_id,
            "num_cells": torch.tensor([target.sum().item()], dtype = torch.float32),
        }
    
    
    def _image_to_tensor(self, image):
        image = np.transpose(image, (2, 0, 1))              #HWC -> CHW
        image = torch.from_numpy(image).float() / 255.0
        return image
    
    
    def _collect_ids(self):
        image_ids = {
            p.stem for p in self.path_to_images.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        }

        label_ids = {
            p.stem for p in self.path_to_labels.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
         }

        common_ids = sorted(image_ids & label_ids, key = lambda x: int(x) if x.isdigit() else x)
        
        only_images = sorted(image_ids - label_ids)
        only_labels = sorted(label_ids - image_ids)

        if only_images:
            print(f"Warning: {len(only_images)} image-файлов без label пропущены.")
        if only_labels:
            print(f"Warning: {len(only_labels)} label-файлов без image пропущены.")

        return common_ids
    
    
    def _read_points(self, filename):
        label_path = self.path_to_labels / f"{filename}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(f"Не найден label-файл: {label_path}")

        if label_path.stat().st_size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        try:
            points = np.loadtxt(label_path, dtype = np.float32)
        except ValueError as e:
            raise ValueError(f"Ошибка чтения {label_path}: файл имеет некорректную структуру") from e

        if points.ndim == 1:                                     #случай одной строки в файле
            if points.size != 3:
                raise ValueError(f"Некорректный формат файла {label_path}")
            points = points[None, :]

        if points.shape[1] != 3:
            raise ValueError(f"Ожидался формат [N, 3], получено {points.shape}")

        return points
    
    
    def _points_to_target(self, points):
        target = np.zeros(self.num_classes, dtype = np.float32)

        if len(points) == 0:
            return target

        class_ids = points[:, 2].astype(np.int64)

        if np.any(class_ids < 0) or np.any(class_ids >= self.num_classes):
            raise ValueError("В разметке для точки задан неверный класс")

        target[:] = np.bincount(class_ids, minlength = self.num_classes).astype(np.float32)

        return target
    

    def _resolve_image_path(self, sample_id):
        for ext in self.IMG_EXTENSIONS:
            path = self.path_to_images / f"{sample_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Не найдено изображение для sample_id={sample_id}")
    

    def _infer_num_classes(self):
        class_ids_set = set()

        for sample_id in self._collect_ids():
            points = self._read_points(sample_id)
            if len(points) == 0:
                continue

            class_ids = points[:, 2].astype(np.int64)
            class_ids_set.update(class_ids.tolist())

        return len(class_ids_set)
    

    def describe(self):
        targets         = self.get_target_matrix()
        total_cells     = targets.sum(axis = 1)
        cells_per_class = targets.sum(axis = 0)

        return {
            "num_samples":          len(self.samples),
            "num_classes":          int(self.num_classes),
            "cells_per_class":      cells_per_class.astype(int).tolist(),
            "mean_cells_per_image": float(total_cells.mean()),
            "std_cells_per_image":  float(total_cells.std()),                       #стандартное отклонение
            "min_cells_per_image":  int(total_cells.min()),
            "max_cells_per_image":  int(total_cells.max()),
        }


    def get_target_matrix(self):
        targets = np.stack([target for _, target in self.samples], axis=0)
        return torch.tensor(targets, dtype = torch.float32)


    
dataset = CellCountDataset(
    root_path = DATASET_ROOT,
    transform = None,
    num_classes = NUM_CLASSES         # 3
)

print(dataset.len_dataset)
print(len(dataset))
print(dataset[0][1])
print(listdir(dataset.path_to_images))