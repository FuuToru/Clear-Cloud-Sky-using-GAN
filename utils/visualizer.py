import os
import torchvision.utils as vutils

def save_generated_images(images, epoch, output_path):
    os.makedirs(f"{output_path}/images", exist_ok=True)
    vutils.save_image(images, f"{output_path}/images/fake_epoch_{epoch}.png", normalize=True)

# ========== utils/metrics.py ==========
# Placeholder for future metric functions (e.g., PSNR, SSIM)
def compute_psnr(img1, img2):
    pass

def compute_ssim(img1, img2):
    pass

# ========== scripts/inference.py ==========
# You can later implement inference to load generator and generate from noise or clean specific input
print("TODO: Implement inference script")