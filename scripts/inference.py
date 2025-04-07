import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from models.generator import Generator
from utils.visualizer import save_output_image

# ==============================
# Argument Parser
# ==============================
def get_args():
    parser = argparse.ArgumentParser(description="Inference: Cloud Removal with GAN")
    parser.add_argument('--input_dir', type=str, default='data/processed/', help='Path to input images')
    parser.add_argument('--output_dir', type=str, default='outputs/images/', help='Path to save output images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/generator.pth', help='Path to trained generator weights')
    parser.add_argument('--image_size', type=int, default=256, help='Size of input images (square)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

# ==============================
# Image Transform
# ==============================
def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # assuming RGB [-1, 1]
    ])

# ==============================
# Inference Function
# ==============================
@torch.no_grad()
def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load generator
    generator = Generator().to(args.device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    generator.eval()

    transform = get_transform(args.image_size)
    
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in tqdm(image_files, desc="Running inference"):
        img_path = os.path.join(args.input_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(args.device)

        # Run Generator
        fake_clear = generator(input_tensor)

        # Save output image
        save_output_image(fake_clear, os.path.join(args.output_dir, img_name))

    print(f"\n✅ Inference completed! Results saved to {args.output_dir}")

# ==============================
# Entry Point
# ==============================
if __name__ == '__main__':
    args = get_args()
    run_inference(args)
