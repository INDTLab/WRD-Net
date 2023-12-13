import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import os
import numpy as np
import random
import time
from PIL import Image
import math
from pavit import Mlp,Conv2d_BN,DWConv2d_BN,DWCPatchEmbed,Patch_Embed_stage,ConvPosEnc,ConvRelPosEnc,FactorAtt_ConvRelPosEnc,MHCABlock,MHCAEncoder,ResBlock,MHCA_stage
from pavit import dpr_generator
from dsntnn import *
from functools import partial
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import einsum
from skimage import transform,data
torch.manual_seed(42)

class Model(nn.Module):
    def __init__(self,
            num_stages=4,
            num_path=[4,4,4,4],
            num_layers=[1,1,1,1],
            embed_dims=[64, 128, 256, 512],
            mlp_ratios=[8, 8, 4, 4],
            num_heads=[8, 8, 8, 8],
            drop_path_rate=0.0,
            num_classes=1000,
            in_channels=3,
            out_channels=20,
            **kwargs,):
        super(Model,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)
        # Encoder_Stem
        self.stem = nn.Sequential(
            Conv2d_BN(
                in_channels,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=1,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=1,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )
        # Encoder_ASP(Atrous Spatial Pyramid)
        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=True,
            ) for idx in range(self.num_stages)
        ])
        # Encoder_PAViT(Parallel Attention Vision Transformer)
        self.mhca_stages = nn.ModuleList([
            MHCA_stage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
            ) for idx in range(self.num_stages)
        ])
        #Decoder_ASP(Atrous Spatial Pyramid)
        self.decoder_patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=False,
            ) for idx in range(self.num_stages)
        ])
        #Decoder_PAViT(Parallel Attention Vision Transformer)
        self.decoder_mhca_stages = nn.ModuleList([
            MHCA_stage(
                embed_dims[idx],
                embed_dims[idx - 1]
                if not (idx - 1) == -1 else embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
            ) for idx in range(self.num_stages)
        ])
        # Decoder_conv
        self.decoder_stem = nn.Conv2d(embed_dims[0],out_channels,1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x):
        # 1. Run the images through our WRD-Net
        embed = []
        x = self.stem(x)  # Shape : [B, C, H/4, W/4]
        embed.append(x)
        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            x = self.mhca_stages[idx](att_inputs)
            embed.append(x)
        embed = embed[::-1]

        for idx in range(self.num_stages):
            if idx == 0:
                x = F.interpolate(embed[idx], size=(38, 50), mode='bilinear', align_corners=True)
            if idx == 1:
                x = F.interpolate(x, size=(75, 100), mode='bilinear', align_corners=True)
            if idx == 2:
                x = F.interpolate(x, size=(150, 200), mode='bilinear', align_corners=True)
            if idx == 3:
                x = F.interpolate(x, size=(300, 400), mode='bilinear', align_corners=True)
            decoder_att_inputs = self.decoder_patch_embed_stages[self.num_stages - 1 - idx](x)
            x = self.decoder_mhca_stages[self.num_stages - 1 - idx](decoder_att_inputs)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnorm_heatmaps = self.decoder_stem(x)
        # 3. Normalize the heatmaps
        heatmaps = flat_softmax(unnorm_heatmaps)
        # 4. Calculate the coordinates
        coords = dsnt(heatmaps)

        return coords, heatmaps


def get_file(img_path,label_path):
    img_lst = os.listdir(img_path)
    data_lst = []
    for img in img_lst:
        img_name = img[:-4]
        lab = img_name + '.npy'
        img_file_path = os.path.join(img_path,img)
        label_file_path = os.path.join(label_path,lab)
        data_lst.append((img_file_path,label_file_path,img_name))
    return data_lst

class JointsDataSet(Dataset):
    def __init__(self,data_lst,data_augment=None):
        self.data_lst = data_lst
        self.data_augment = data_augment
        print(f'got {len(data_lst)} images and ground-truths')

    def __getitem__(self,index):
        img = Image.open(self.data_lst[index][0])
        lab = np.load(self.data_lst[index][1])
        name = self.data_lst[index][2]
        img = TF.to_tensor(img).div(255)
        lab = torch.from_numpy(lab)
        image_size = [img.shape[2],img.shape[1]]#[400,300]
        lab_tensor = (lab * 2 + 1) / torch.Tensor(image_size) - 1
        if self.data_augment:
            img,lab_tensor = self.data_augment(img,lab_tensor)
        return img,lab_tensor,name

    def __len__(self):
        return len(self.data_lst)


img_root_path = './data/WRSD/images'
label_root_path = './data/WRSD/annotations'

data_lst = get_file(img_root_path,label_root_path)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

lr = 5e-4
weight_decay = 1e-4
train_bs = 2
num_epochs = 100

train_mse_lst = []
val_mse_lst = []

evaluate_interval = 20
save_interval = 50
# Save path for models
snapshot_path = './results/model/'
# Save path for detection results
heatmap_path = './results/coordsresult/'
# Save path for loss arrays
log_path = './results/loss/'
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
if not os.path.exists(heatmap_path):
    os.makedirs(heatmap_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
best_val_loss = 100.0
num_joints=20
HeatMapPoints=np.zeros((num_joints,2),dtype=np.float32)
kf = KFold(n_splits = 3)
for j,(train_index,val_index) in enumerate(kf.split(data_lst)):
    net = Model(num_stages=4,
            num_path=[3,3,3,3],
            num_layers=[1,1,1,1],
            embed_dims=[64, 128, 256, 512],
            mlp_ratios=[4, 4, 4, 4],
            num_heads=[8, 8, 8, 8],
            drop_path_rate=0.2,
            num_classes=1000,
            in_channels=3,
            out_channels=20).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_lst = [data_lst[ii] for ii in train_index]
    val_lst = [data_lst[ii] for ii in val_index]
    train_set = JointsDataSet(train_lst)
    val_set = JointsDataSet(val_lst)
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for epoch in range(num_epochs):
        train_mse = 0
        tic = time.time()
        net.train()
        for img,target_var,name in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            target_var = target_var.to(device)
            coords, heatmaps = net(img)
            # Per-location euclidean losses
            euc_losses = euclidean_losses(coords, target_var)
            # Per-location regularization losses
            reg_losses = js_reg_losses(heatmaps, target_var, sigma_t=1.0)
            # Per-location focal losses
            focal_losses = focal_reg_losses(heatmaps, target_var, sigma_t=1.0,gamma = 1)
            # Combine losses into an overall loss
            loss = average_loss(euc_losses + reg_losses + focal_losses)
            loss.backward()
            optimizer.step()
            train_mse += loss.item()*len(img)
        toc = time.time()
        mean_loss = train_mse/len(train_set)
        train_mse_lst.append(mean_loss)
        np.save(log_path + f'{j}_train_mse.npy',np.array(train_mse_lst))

        print(f'{j} flod:[{epoch+1}]/[{num_epochs}] train loss:{mean_loss} time consumption:{toc-tic}s')
        if (epoch+1) % evaluate_interval == 0:
            print(f'{j} flod:evaluate model on validation set......')
            net.eval()
            val_mse = 0
            with torch.no_grad():
                for img,target_var,name in val_loader:
                    img = img.to(device)
                    target_var = target_var.to(device)
                    coords, heatmaps = net(img)
                    imgsize_tensor = torch.Tensor([img.shape[3],img.shape[2]]).to(device)#[400,300]
                    save_coords = ((coords + 1) * imgsize_tensor - 1) / 2
                    # Per-location euclidean losses
                    euc_losses = euclidean_losses(coords, target_var)
                    # Per-location regularization losses
                    reg_losses = js_reg_losses(heatmaps, target_var, sigma_t=1.0)
                    # Per-location focal losses
                    focal_losses = focal_reg_losses(heatmaps, target_var, sigma_t=1.0,gamma = 1)
                    # Combine losses into an overall loss
                    val_loss = average_loss(euc_losses + reg_losses + focal_losses)
                    val_mse += val_loss.item()
                    # Save results on test set
                    name = ''.join(name)
                    if (epoch+1) % num_epochs == 0:
                        np.save(heatmap_path+f'val_{epoch+1}_{name}.npy',save_coords.cpu().numpy()[0])
            mean_val_loss = val_mse/len(val_set)
            val_mse_lst.append(mean_val_loss)
            np.save(log_path + f'{j}_val_mse.npy',np.array(val_mse_lst))

            
            print(f'{j} flod:loss on val set{mean_val_loss}')
            # Save best model on test set
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
                torch.save(state_dict, snapshot_path + f'{j}_best_model.pth')
                print(f'{j} flod:save best val model at epoch{epoch+1}')

        if (epoch+1) % save_interval == 0:
            # Save the model every save_interval
            state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state_dict, snapshot_path + f'{j}_model_epoch{epoch+1}.pth')
