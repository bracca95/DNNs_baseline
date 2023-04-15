import os
import torch

from PIL import Image
from glob import glob
from typing import Optional, List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.config_parser import Config
from src.tools import Logger


class DefectViews(Dataset):

    label_to_idx = {
        "bubble": 0, 
        "point": 1,
        "break": 2,
        "dirt": 3,
        "mark": 4,
        "scratch": 5
    }

    def __init__(self, dataset_path: str, crop_size: int, img_size: Optional[int] = None):
        self.dataset_path: str = dataset_path
        self.image_list: Optional[List[str]] = self.get_image_list()
        self.label_list: Optional[List[int]] = self.get_label_list()

        self.crop_size = crop_size
        self.img_size = img_size

        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def get_image_list(self):
        self.image_list = [f for f in glob(os.path.join(self.dataset_path, "*.png"))]
        
        if not all(map(lambda x: x.endswith(".png"), self.image_list)) or self.image_list == []:
            raise ValueError("incorrect image list. Check the provided path for your dataset.")

    def get_label_list(self):
        if self.image_list is None:
            self.get_image_list()

        filenames = list(map(lambda x: os.path.basename(x), self.image_list))
        label_list = list(map(lambda x: x.rsplit("_")[0], filenames))
        self.label_list = [DefectViews.label_to_idx[defect] for defect in label_list]

    def load_image(self, path: str) -> torch.Tensor:
        img_pil = Image.open(path).convert("L")
        
        # rescale [0-255](int) to [0-1](float)
        totensor = transforms.Compose([transforms.ToTensor()])
        img = totensor(img_pil)

        # crop
        if img_pil.size[0] * img_pil.size[1] < self.crop_size * self.crop_size:
            m = min(img_pil.size)
            centercrop = transforms.Compose([transforms.CenterCrop((m, m))])
            resize = transforms.Compose([transforms.Resize((self.crop_size, self.crop_size))])
            
            img = centercrop(img)
            img = resize(img)

            Logger.instance().debug(f"image size for {os.path.basename(path)} is less than required. Upscaling.")
        else:
            centercrop = transforms.Compose([transforms.CenterCrop((self.crop_size, self.crop_size))])
            img = centercrop(img)
        
        # resize (if required)
        if self.img_size is not None:
            resize = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])
            img = resize(img)

        # normalize
        if self.mean is not None and self.std is not None:
            normalize = transforms.Normalize(self.mean, self.std)
            img = normalize(img)

        return img # type: ignore

    @staticmethod
    def compute_mean_std(dataset: Dataset, config: Config):
        # https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/31
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        mean = 0.0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(dataloader.dataset)

        var = 0.0
        pixel_count = 0
        for batch in dataloader:
            images, _ = batch
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
            pixel_count += images.nelement()
        std = torch.sqrt(var / pixel_count)

        if any(map(lambda x: torch.isnan(x), mean)) or any(map(lambda x: torch.isnan(x), std)):
            raise ValueError("mean or std are none")

        config.dataset_mean = mean.tolist()
        config.dataset_std = std.tolist()
        config.serialize(os.getcwd(), "config/config.json")
        Logger.instance().warning(f"Mean: {mean}, std: {std}. Run the program again.")

    def __getitem__(self, index):
        curr_img_batch = self.image_list[index]

        # there is a bug for which label list are not read
        if self.label_list is None:
            self.get_label_list()
        
        return self.load_image(curr_img_batch), self.label_list[index]

    def __len__(self):
        return len(self.image_list) # type: ignore
