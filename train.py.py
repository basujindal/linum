
from contextlib import nullcontext
import os
import torch
import numpy as np
from datetime import datetime
import time
from prettytable import PrettyTable
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from img_utils import *
import wandb
import torch
import torch.nn as nn


class Timer:
    def __init__(self, start_msg = "", end_msg = ""):
    
        self.start_msg = start_msg
        self.end_msg = end_msg
        
    def __enter__(self):
        if self.start_msg != "":
            print(self.start_msg)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(self.end_msg, f"{elapsed_time:.3f} sec")


def count_parameters(model, print_table = False):
    
    total_params = 0
    
    if(print_table):
        table = PrettyTable(["Modules", "Parameters", "dtype", "Required Grad", "Device"]) 
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        
        if(print_table):
            table.add_row([name, parameter.shape, parameter.dtype, parameter.requires_grad, parameter.device ])
            
        total_params += params
        
    if(print_table):
        print(table)
        
    if total_params/1e9 > 1:
        print(f"Total Trainable Params: {total_params/1e9} B")
    else:
        print(f"Total Trainable Params: {total_params/1e6} M")
        
    return total_params



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
            mask = self.maskTransform(mask)
        
        return image, src,mask

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image to [0, 1]
])
maskTransform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
])


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

        self.decoders = nn.ModuleList([Decoder(*i) for i in [(512,256, (21,37)), (256,128,(42,74)), (128, 64,(84,149)), (64, 32,(168,298))]])

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

        for idx, dec in enumerate(self.decoders):
            x = dec(x, skips[3-idx])

        image = self.tanh(self.imgDecoder(x))
        mask = self.maskDecoder(x)
        return image, self.sigmoid(mask)


lr = 1e-4
# bce = nn.BCEWithLogitsLoss()
bce = nn.BCELoss()
bs = 64


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
unet = Unet().to(device)

optimizer = torch.optim.AdamW(unet.parameters(), lr)

image_dir = '/root/data/linum/train/corrupted_imgs/'
mask_dir = '/root/data/linum/train/binary_masks/'
src_dir = '/root/data/linum/train/src_imgs/'
nparams = count_parameters(unet, print_table=False)
dataset = ImageMaskDataset(image_dir=image_dir, mask_dir=mask_dir, src_dir = src_dir,transform=transform, maskTransform = maskTransform)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=24)



log = False
log = True
log_iter = 10
img_losses = 0
mask_losses = 0
iter = 0
epochs = 25

if log:
    config={"epochs": epochs, "batch_size": bs,"lr": lr}
    wandb.init(project='linum', entity='basujindal123', config=config)


# convert_img_tensor_to_pil_img((masks.bool()*images)[0])
# convert_img_tensor_to_pil_img((images)[0])
# # convert_mask_tensor_to_pil_img(masks[0])
# unet = torch.compile(unet)


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

        img_loss = 2*torch.sum(torch.abs(masks.bool()*(img_pred-src)))/m
        mask_loss = 100*bce(mask_pred, masks)
        loss = mask_loss + img_loss

        loss.backward()
        optimizer.step()

        img_losses+=img_loss.item()
        mask_losses+=mask_loss.item()

        if (iter+1)%log_iter == 0:

            if log:
                wandb.log({
                    'loss': (mask_losses+img_losses)/log_iter,
                    'mask_loss': mask_losses/log_iter,
                    'img_loss': img_losses/log_iter,
                    'Corrupted Images': [wandb.Image(i) for i in images[:4].detach()],
                    'Reconstructed Images' : [wandb.Image(i) for i in img_pred[:4].detach()],
                    'Src Images' : [wandb.Image(i) for i in src[:4].detach()],
                    'Masks' : [wandb.Image(i) for i in masks[:4].detach()],
                    'Predicted Masks' : [wandb.Image(i) for i in (mask_pred[:4]).detach()],
                    })

            # print(epoch, iter, (mask_losses+img_losses)/log_iter, mask_losses/log_iter,img_losses/log_iter)
            mask_losses = 0
            img_losses = 0








