# Clear Cloud Sky using GAN

**Clear Cloud Sky using GAN** is a project that employs a Generative Adversarial Network (GAN) to remove clouds from satellite images, enhancing their clarity. Built using the EuroSAT dataset, this project provides tools for training a GAN model and performing inference to generate cloud-free images.

## Features
- Train a GAN model to remove clouds from satellite imagery.
- Perform inference on new images using a pre-trained generator.
- Modular design with configurable hyperparameters.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (GPU support recommended)
- Additional dependencies listed in `requirements.txt`

### Setup
Clone the repository and install the required packages:
```bash
git clone https://github.com/your-username/fuutoru-gan-clear-cloud.git
cd fuutoru-gan-clear-cloud
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Download the EuroSAT dataset from [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) and place it in the `data/processed/` directory. Refer to `datasource/readme.txt` for more details.

### 2. Training the Model
Run the training script with the provided configuration:
```bash
python scripts/train.py --config config/train_config.yaml
```
- Modify `config/train_config.yaml` to adjust hyperparameters like `batch_size`, `num_epochs`, or `dataset_path`.

### 3. Running Inference
Generate cloud-free images using a trained model:
```bash
python scripts/inference.py --input_dir data/processed/ --output_dir outputs/images/ --checkpoint checkpoints/generator.pth
```
- `--input_dir`: Path to cloudy images.
- `--output_dir`: Path to save generated images.
- `--checkpoint`: Path to the trained generator weights.

## GAN Architecture
The GAN model in this project consists of two main components: the **Generator** and the **Discriminator**, designed to work together to remove clouds from satellite images.

- **Generator**: Takes a latent vector as input and generates a synthetic cloud-free image. It uses transposed convolutional layers to upsample the input into a 64x64 RGB image.
- **Discriminator**: Evaluates whether an image is real (cloud-free) or fake (generated), using convolutional layers to downsample and classify the input.
- **Loss Functions**: Binary Cross-Entropy (BCE) loss is used to train both networks adversarially.

The architecture is illustrated below:

![GAN Architecture](/assets/GAN.png)

## Directory Structure
```
fuutoru-gan-clear-cloud/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── config/                     # Configuration files
│   └── train_config.yaml       # Training hyperparameters
├── datasets/                   # Dataset handling
│   └── sky_dataset.py          # Custom dataset loader for EuroSAT
├── datasource/                 # Data source information
│   └── readme.txt              # Instructions for downloading EuroSAT dataset
├── models/                     # GAN components
│   ├── discriminator.py        # Discriminator model
│   ├── generator.py            # Generator model
│   └── losses.py               # Loss functions for GAN training
├── scripts/                    # Executable scripts
│   ├── inference.py            # Inference script for cloud removal
│   └── train.py                # Training script for the GAN
└── utils/                      # Utility functions
    ├── logger.py               # Logging training metrics
    ├── metrics.py              # Placeholder for evaluation metrics
    └── visualizer.py           # Tools for saving and visualizing images
```

## Configuration
The `train_config.yaml` file contains key training parameters:
- `batch_size`: 64
- `image_size`: 64
- `num_epochs`: 100
- `learning_rate`: 0.0002
- `latent_dim`: 100
- `device`: cuda (or cpu if no GPU is available)

Adjust these settings based on your hardware and requirements.

## Results
- Training logs are saved in `outputs/metrics/log.txt`.
- Generated images are stored in `outputs/images/` during training and inference.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, feel free to open an issue or contact [huutri231103@gmail.com](mailto:huutri231103@gmail.com).