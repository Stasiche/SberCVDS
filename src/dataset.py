import PIL
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset
from fastai.data.external import untar_data, URLs
import csv
from os.path import join


def get_dataset(file_path, mode):
    with open(join(file_path, 'noisy_imagewoof.csv'), 'r') as f:
        reader = csv.reader(f)
        _ = next(reader)
        return [row[:2] for row in reader if row[0].startswith(mode)]


class ImageWoofDataset(Dataset):
    __breads_map = {el: i for i, el in enumerate(['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754',
                                                  'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641'])}

    def __init__(self, mode):
        self.datapath = untar_data(URLs.IMAGEWOOF)
        self.data = get_dataset(self.datapath, mode)
        self.transforms = Compose([Resize((256, 256)),
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_subpath, label_str = self.data[idx]
        image = PIL.Image.open(join(self.datapath, img_subpath))
        # в датасете есть черно-белые изображения (87 и 11 в трейне и тесте соотвественно). это около процента данных
        image = image.convert('RGB')
        image = self.transforms(image)
        label = self.__breads_map[label_str]
        return image, label
