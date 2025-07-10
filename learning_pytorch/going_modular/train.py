from cv2 import transform
import torch
from torchvision import transforms
from pathlib import Path
import data_setup, engine, model_builder, utils
import torch.nn as nn

BATCH_SIZE = 32
EPOCH = 3
HIDDEN_UNITS = 10
LEARNING_RATE = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "learning_pytorch/data/pizza_steak_sushi/train"
test_dir = "learning_pytorch/data/pizza_steak_sushi/test"

data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
train_dataloader, test_dataloader, classes = data_setup.create_dataloader(
                                                train_dir = train_dir,
                                                test_dir= test_dir,
                                                transforms=data_transform,
                                                batch_size=BATCH_SIZE)

model_TinyVGG = model_builder.TinyVGG(in_channels=3, 
                                      out_shape=len(classes),
                                      hidden_units=HIDDEN_UNITS).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_TinyVGG.parameters(),
                             lr = LEARNING_RATE)

engine.train(model = model_TinyVGG,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=EPOCH,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             device = device)

utils.save_model(model=model_TinyVGG,
                 target_dir="learning_pytorch/models",
                 model_name="modular_TinyVGG.pth")