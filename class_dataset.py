from torch.utils.data import Dataset
from os import listdir
from pathlib import Path

#DATASET_ROOT = "/content/drive/MyDrive/Colab Notebooks/HistoCellCount/dataset_small"
DATASET_ROOT = "/home/terenkovkirill/HistoCellCount/dataset_small"

class CellCountDataset(Dataset):
    def __init__(self, root_path, num_classes = None, transform = None):
        self.transform =  transform
        self.num_classes = num_classes
        self.root_path = Path(root_path)

        if root_path is not None:
            path_to_images = Path(root_path + "/images")
            path_to_labels = Path(root_path + "/labels")
        else:
            raise ValueError("Не передан путь к корневой папке.")

        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        
        self.ids = []
        for file in listdir(path_to_images):
            if Path(file).suffix.lower() == ".png":
                self.ids.append(file.stem)

        self.len_dataset = len(self.list_name_file)

    def __len__(self):
        return self.len_dataset
    
dataset = CellCountDataset(
    root_path = DATASET_ROOT,
    transform = None,
    num_classes = None          # 3
)