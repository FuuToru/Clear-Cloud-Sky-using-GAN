import os
import torchvision.utils as vutils, save_image

def save_generated_images(images, epoch, output_path):
    os.makedirs(f"{output_path}/images", exist_ok=True)
    vutils.save_image(images, f"{output_path}/images/fake_epoch_{epoch}.png", normalize=True)

def save_output_image(tensor, path):
    # tensor shape: (1, C, H, W)
    tensor = tensor.clone().detach().cpu()
    tensor = (tensor + 1) / 2  # [-1,1] => [0,1]
    save_image(tensor, path)



