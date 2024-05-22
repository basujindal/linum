# %%
import os
import torch
import numpy as np
import time
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import argparse



class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, src_dir,transform=None, maskTransform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.src_dir = src_dir
        self.transform = transform
        self.maskTransform = maskTransform
        self.image_names = sorted(os.listdir(image_dir))
        self.src_names = sorted(os.listdir(src_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        mask_name = self.mask_names[idx]
        src_name = self.src_names[idx]
        
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        src_path = os.path.join(self.src_dir, src_name)
        
        image = Image.open(image_path).convert('RGB')
        src = Image.open(src_path).convert('RGB')
        mask = np.load(mask_path)
        
        
        if self.transform:
            image = self.transform(image)
            src = self.transform(src)
            mask = torch.tensor(mask).unsqueeze(0).float()
        
        return image, src,mask

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize the image to [-1,1]
])

# %%
class ResnetBlock(nn.Module):
    def __init__(self, out_c,padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.conv2 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.conv3 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.norm = nn.BatchNorm2d(out_c)
        self.silu = nn.SiLU()

    def forward(self, x):

        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.norm(x) + x
        x = self.silu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_c, out_c,  padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c,out_c,3,1,padding)
        self.conv2 = nn.Conv2d(out_c,out_c,3,1,padding)
        self.conv3 = nn.Conv2d(out_c,out_c,3,1,padding)

        self.resBlocks = nn.ModuleList([ResnetBlock(out_c, padding) for i in range(3)])

        self.silu = nn.SiLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.silu(x)

        for block in self.resBlocks:
            x = block(x)

        return self.pool(x), x


class Decoder(nn.Module):
    def __init__(self, in_c, out_c, img_sizes, padding=1):
        super().__init__()

        self.upsample = nn.Upsample(size=img_sizes)


        self.conv1 = nn.Conv2d(in_c,in_c,3,1, padding)
        self.conv2 = nn.Conv2d(in_c,in_c,3,1, padding)
        self.conv3 = nn.Conv2d(in_c,in_c//2,3,1, padding)


        self.conv4 = nn.Conv2d(in_c,out_c,3,1, padding)
        self.conv5 = nn.Conv2d(out_c,out_c,3,1, padding)
        self.conv6 = nn.Conv2d(out_c,out_c,3,1, padding)


        self.resBlocks = nn.ModuleList([ResnetBlock(out_c, padding) for i in range(3)])

        self.silu = nn.SiLU()

    def forward(self, x, skip):

        x = self.upsample(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.cat((x, skip), dim = 1)
        x = self.conv6(self.conv5(self.conv4(x)))
        x = self.silu(x)

        for block in self.resBlocks:

            x = block(x)

        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([Encoder(*i) for i in [(3,32), (32,64), (64, 128), (128, 256), (256, 512)]])

        self.imageDecoders = nn.ModuleList([Decoder(*i) for i in [(512,256, (21,37)), (256,128,(42,74)), (128, 64,(84,149)), (64, 32,(168,298))]])

        self.maskDecoders = nn.ModuleList([Decoder(*i) for i in [(512,256, (21,37)), (256,128,(42,74)), (128, 64,(84,149)), (64, 32,(168,298))]])

        self.imgDecoder = nn.Conv2d(32,3,1,1)
        self.tanh = nn.Tanh()

        self.maskDecoder = nn.Conv2d(32,1,1,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = skips[-1]

        for idx, dec in enumerate(self.imageDecoders):
            x = dec(x, skips[3-idx])

        image = self.tanh(self.imgDecoder(x))

        x = skips[-1]

        for idx, dec in enumerate(self.maskDecoders):
            x = dec(x, skips[3-idx])
            
        mask = self.sigmoid(self.maskDecoder(x))
        return image, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--bs', type=int, default=80, help='Batch size (works for 80GB fp32 training)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log', action='store_true', help='Enable logging')
    parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
    parser.add_argument('--log_iter', type=int, default=20, help='Log after n iterations')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')

    args = parser.parse_args()
    bs = args.bs
    lr = args.lr
    log = args.log
    epochs = args.epochs
    log_iter = args.log_iter
    data_dir = args.data_dir


    if log:
        config={"epochs": epochs, "batch_size": bs,"lr": lr}
        wandb.init(project='linum', entity='basujindal123', config=config)
    # %%
    image_dir = os.path.join(data_dir,'train/corrupted_imgs/')
    mask_dir = os.path.join(data_dir,'train/binary_masks/')
    src_dir = os.path.join(data_dir,'train/src_imgs/')
    dataset = ImageMaskDataset(image_dir=image_dir, mask_dir=mask_dir, src_dir = src_dir,transform=transform, maskTransform = None)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=24)


    # %% [markdown]
    # ## Train

    # %%
    bce = nn.BCELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = Unet().to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr)

    # %%
    img_losses = 0
    mask_losses = 0
    iter = 0
    num_imgs = 3

    # %%
    for epoch in range(epochs):

        for images, src, masks in tqdm(dataloader):

            unet.train()
        
            images = images.to(device)
            masks = masks.to(device)
            src = src.to(device)
            m = torch.sum(masks)
            unet.zero_grad()

            iter+=1
            img_pred, mask_pred = unet(images)

            img_loss = 2*torch.sum(torch.abs((masks.bool() == 1)*img_pred-(masks.bool() == 1)*src))/m
            mask_loss = bce(mask_pred, masks)
            loss = (mask_loss + img_loss)

            loss.backward()

            optimizer.step()
            img_losses+=img_loss.item()
            mask_losses+=mask_loss.item()

            if (iter+1)%log_iter == 0:

                if log:
                    combined_images = (masks[:num_imgs].bool())*img_pred[:num_imgs].detach() + (masks[:num_imgs].bool() != 1)*images[:num_imgs]

                    wandb.log({
                        'loss': (mask_losses+img_losses)/log_iter,
                        'mask_loss': mask_losses/log_iter,
                        'img_loss': img_losses/log_iter,
                        'Corrupted Images': [wandb.Image(i) for i in images[:num_imgs].detach()],
                        'Reconstructed Images' : [wandb.Image(i) for i in img_pred[:num_imgs].detach()],
                        'Combined Images' : [wandb.Image(i) for i in combined_images],
                        'Masks' : [wandb.Image(i) for i in masks[:num_imgs].detach()],
                        'Predicted Masks' : [wandb.Image(i) for i in (mask_pred[:num_imgs]).detach()],
                        })

                print(epoch, iter, (mask_losses+img_losses)/log_iter, mask_losses/log_iter,img_losses/log_iter)
                mask_losses = 0
                img_losses = 0

        print("Saving")
        torch.save(unet.state_dict(), "unet_" + str(epoch) + ".pth")

    # %% [markdown]
    # ## Validation

    print("Validating the model")
    # %%
    bs = 25
    image_dir = os.path.join(data_dir,'validation/corrupted_imgs/')
    mask_dir = os.path.join(data_dir,'validation/binary_masks/')
    src_dir = os.path.join(data_dir,'validation/src_imgs/')
    val_dataset = ImageMaskDataset(image_dir=image_dir, mask_dir=mask_dir, src_dir = src_dir,transform=transform, maskTransform = None)
    valloader = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=25)

    # %%
    img_losses = 0
    mask_losses = 0

    with torch.no_grad():

        for images, src, masks in tqdm(valloader):

            unet.eval()
        
            images = images.to(device)
            masks = masks.to(device)
            src = src.to(device)
            m = torch.sum(masks)

            img_pred, mask_pred = unet(images)
            img_loss = 2*torch.sum(torch.abs((masks.bool() == 1)*img_pred-(masks.bool() == 1)*src))/m
            mask_loss = bce(mask_pred, masks)
            loss = (mask_loss + img_loss)

            img_losses+=img_loss.item()
            mask_losses+=mask_loss.item()

            if log:
                combined_images = (masks.bool())*img_pred + (masks.bool() != 1)*images

                wandb.log({
                    'Validation Corrupted Images': [wandb.Image(i) for i in images],
                    'Validation Reconstructed Images' : [wandb.Image(i) for i in img_pred],
                    'Validation Combined Images' : [wandb.Image(i) for i in combined_images],
                    'Validation Masks' : [wandb.Image(i) for i in masks],
                    'Validation Predicted Masks' : [wandb.Image(i) for i in mask_pred],
                    })

        if log:
            wandb.log({
                'Validation loss': (mask_losses+img_losses/len(valloader)),
                'Validation mask_loss': mask_losses/len(valloader),
                'Validation img_loss': img_losses/len(valloader),
                })

        print((mask_losses+img_losses)/len(valloader), mask_losses/len(valloader), img_losses/len(valloader))
