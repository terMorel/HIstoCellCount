from torch.utils.data import Dataset
from os import listdir
from pathlib import Path
import cv2

#DATASET_ROOT = "/content/drive/MyDrive/Colab Notebooks/HistoCellCount/dataset_small"
DATASET_ROOT = "/home/terenkovkirill/HistoCellCount/dataset_small"
NUM_CLASSES = 3

class CellCountDataset(Dataset):
    def __init__(self, root_path, num_classes, transform = None):
        self.transform =  transform
        self.num_classes = num_classes

        if root_path is not None:
            self.root_path = Path(root_path)
            self.path_to_images = self.root_path / "images"
            self.path_to_labels = self.root_path / "labels"
        else:
            raise ValueError("Не передан путь к корневой папке.")
        
        ids = []
        for file in listdir(self.path_to_images):
            file = Path(file)
            if file.suffix.lower() == ".png":
                ids.append(file.stem)

        self.count_cells = []

        for filename in ids:
            try:
                with open (self.path_to_labels / f"{filename}.txt") as file:
                    lines = file.readlines()

                    count_cells = [0] * self.num_classes
                    for line in lines:
                        class_num = int(line.split()[2])
                        count_cells[class_num] += 1
                        
                    self.count_cells.append((filename, count_cells))
                        
            except FileNotFoundError:
                print(f"Файл {filename}.txt не найден")
            except Exception as e:
                print(f"Произошла ошибка: {e}")
            
            self.len_dataset = len(self.count_cells)

    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, index):
        if index < 0 or index >= self.len_dataset:
            raise ValueError("Ошибочно задан индекс.")
        
        image_path = self.path_to_images / f"{self.count_cells[index][0]}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, self.count_cells[index][1]
    
dataset = CellCountDataset(
    root_path = DATASET_ROOT,
    transform = None,
    num_classes = NUM_CLASSES         # 3
)

print(len(dataset))
print(dataset[0][1])