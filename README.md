# Fuutoru - Clear Cloud Sky using GAN

This project uses a GAN model to remove clouds from satellite images using the EuroSAT dataset.

## Usage

```bash
# Train the model
python scripts/train.py --config config/train_config.yaml

# Run inference
python scripts/inference.py --input path_to_image.jpg --output outputs/images/
```