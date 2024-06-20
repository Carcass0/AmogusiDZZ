import torch
from models import Generator
from config import DEVICE, CHANNELS_IMG, IMAGE_SIZE
from PIL import Image
import numpy as np
from dataset import both_transform, transform_only_input

def load_image(image_path):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    augmentations = both_transform(image=image)
    input_image = augmentations["image"]
    input_image = transform_only_input(image=input_image)["image"]
    input_image = np.expand_dims(input_image, axis=0)
    input_image = torch.from_numpy(input_image).to(DEVICE)
    return input_image

def predict(model, image_path):
    model.eval()
    with torch.no_grad():
        input_image = load_image(image_path)
        output = model(input_image)
        output = output.cpu().numpy().squeeze()
        output = (output * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return output

def main():
    model = Generator(in_channels=CHANNELS_IMG).to(DEVICE)
    checkpoint = torch.load("gen.pth.tar", map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    image_path = "path/to/your/image.jpg"
    output = predict(model, image_path)

    # Отображение результата
    import matplotlib.pyplot as plt
    plt.imshow(output)
    plt.show()

if __name__ == "__main__":
    main()
