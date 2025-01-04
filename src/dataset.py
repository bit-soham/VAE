import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir, dataset_size=None, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if dataset_size is None:
            dataset_size = len(self.image_files)

        self.image_files = random.sample(self.image_files, dataset_size)
        self.transform = transform
        self.image_paths = [os.path.join(self.image_dir, file_name) for file_name in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # Convert from PIL Image to Tensor
        return image


def get_dataloader(image_dir, dataset_size, batch_size, image_size, num_workers=2, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 3-channel normalization
    ])
    
    dataset = ImageDataset(image_dir, dataset_size=dataset_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader
