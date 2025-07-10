
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

NUM_WORKER = os.cpu_count()

def create_dataloader(
    train_dir: str,
    test_dir: str,
    transforms: transforms.Compose,
    batch_size: int,
):
    train_data = datasets.ImageFolder(
        root = train_dir,
        transform = transforms,
        ) 
    test_data = datasets.ImageFolder(
        root = test_dir,
        transform = transforms
    )


    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset= train_data,
        batch_size= batch_size,
        shuffle= True,
        num_workers= NUM_WORKER,
        pin_memory= True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKER,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
