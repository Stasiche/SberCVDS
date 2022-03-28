import PIL
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset
from fastai.data.external import untar_data, URLs
import csv
from os.path import join
from src.breed_convector import BreedConvector


def get_dataset(file_path, mode):
    with open(join(file_path, 'noisy_imagewoof.csv'), 'r') as f:
        reader = csv.reader(f)
        _ = next(reader)
        return [row[:2] for row in reader if row[0].startswith(mode)]


class ImageWoofDataset(Dataset):
    def __init__(self, mode):
        self.datapath = untar_data(URLs.IMAGEWOOF)
        self.data = get_dataset(self.datapath, mode)
        self.transforms = Compose([Resize((256, 256)),
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])

        self.convector = BreedConvector()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_subpath, label_str = self.data[idx]
        image = PIL.Image.open(join(self.datapath, img_subpath))
        # в датасете есть черно-белые изображения (87 и 11 в трейне и тесте соотвественно). это около процента от выборок
        image = image.convert('RGB')
        image = self.transforms(image)
        label = self.convector.dirname_to_index[label_str]
        return image, label
