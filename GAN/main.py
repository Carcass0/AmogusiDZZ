import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import Satellite2Map_Data
from models import Generator, Discriminator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename="log.txt", level=logging.INFO)
logging.debug("Debug logging test...")
torch.backends.cudnn.benchmark = True
Gen_loss = []
Dis_loss = []

def train(netG, netD, train_dl, OptimizerG, OptimizerD, L1_Loss, BCE_Loss):
    for idx, (x,y, y_class) in enumerate(train_dl):
        x = x.cuda()
        y = y.cuda()
        y_class = y_class.float().unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        y_fake = netG(x)
        D_real = netD(x,y)
        D_real_loss = BCE_Loss(D_real, torch.ones_like(D_real))
        D_fake = netD(x,y_fake.detach())
        D_fake_loss = BCE_Loss(D_fake, torch.zeros_like(D_fake))
        D_class_loss = BCE_Loss(D_real, y_class) * config.CLASSIFICATION_LAMBDA
        D_loss = (D_real_loss + D_fake_loss + D_class_loss) / 3

        netD.zero_grad()
        Dis_loss.append(D_loss.item())
        D_loss.backward()
        OptimizerD.step()

        D_fake = netD(x, y_fake)
        G_fake_loss = BCE_Loss(D_fake, torch.ones_like(D_fake))
        L1 = L1_Loss(y_fake,y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        OptimizerG.zero_grad()
        Gen_loss.append(G_loss.item())
        G_loss.backward()
        OptimizerG.step()

def main():
    netD = Discriminator(in_channels=3).cuda()
    netG = Generator(in_channels=3).cuda()
    OptimizerD = torch.optim.Adam(netD.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    OptimizerG = torch.optim.Adam(netG.parameters(),lr=config.LEARNING_RATE,betas=(config.BETA1,0.999))
    BCE_Loss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,netG,OptimizerG,config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,netD,OptimizerD,config.LEARNING_RATE
        )
    train_dataset = Satellite2Map_Data("./dataset/train/images", "./dataset/train/labels")
    train_dl = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
    val_dataset = Satellite2Map_Data("./dataset/val/images", "./dataset/val/labels")
    val_dl = DataLoader(val_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
    for epoch in range(config.NUM_EPOCHS):
        logging.info(f"Epoch {epoch}")
        train(
            netG, netD, train_dl, OptimizerG, OptimizerD, L1_Loss, BCE_Loss
        )
        if config.SAVE_MODEL and epoch % 50 == 0:
            save_checkpoint(netG, OptimizerG, filename=config.CHECKPOINT_GEN)
            save_checkpoint(netD, OptimizerD, filename=config.CHECKPOINT_DISC)
        if epoch % (config.NUM_EPOCHS // 20) == 0:
            save_some_examples(netG, val_dl, epoch, folder="evaluation")
main()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(Gen_loss, label="Generator")
plt.plot(Dis_loss, label="Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("figure1.png")
