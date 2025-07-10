import os
from zipfile import ZipFile
from pathlib import Path
import requests

data_path = Path("learning_pytorch/data")
image_path = data_path/ "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} Directory Already Exists!")
else:
    image_path.mkdir(parents=True, exist_ok= True)
    
    
zip_path = image_path / "pizza_steak_sushi.zip"

if zip_path.is_file():
    print(f"{zip_path} File Already Exists!")
else:
    with open(zip_path, "wb") as f:
        response = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading Zipfile!")
        f.write(response.content)
        
    with ZipFile(zip_path, "r") as f:
        print("Unzipping pizza, steak, sushi....")
        f.extractall(image_path)

os.remove(zip_path)