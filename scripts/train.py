import torch
import yaml
from torch.utils.data import DataLoader
from datasets.sky_dataset import SkyDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.losses import gan_loss
from utils.logger import Logger
from utils.visualizer import save_generated_images

with open("config/train_config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg["device"])

dataset = SkyDataset(cfg["dataset_path"])
dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

G = Generator(cfg["latent_dim"]).to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], 0.999))

logger = Logger()

for epoch in range(cfg["num_epochs"]):
    for imgs in dataloader:
        real = imgs.to(device)
        z = torch.randn(real.size(0), cfg["latent_dim"]).to(device)
        fake = G(z)

        # Train Discriminator
        D_real = D(real)
        D_fake = D(fake.detach())
        loss_D = gan_loss(D_real, True) + gan_loss(D_fake, False)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        D_fake = D(fake)
        loss_G = gan_loss(D_fake, True)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")
    logger.log(epoch, loss_D.item(), loss_G.item())
    save_generated_images(fake, epoch, cfg["output_path"])