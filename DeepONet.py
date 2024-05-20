# from pytorch_lightning.callbacks import ModelCheckpoint
# import pytorch_lightning as pl
import yaml
import argparse
from bisect import bisect
import os
import torch
import shutil
import warnings
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from functools import partial
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torchvision
from sklearn.preprocessing import StandardScaler
from einops import rearrange, repeat, reduce
from mpl_toolkits.axes_grid1 import ImageGrid
import datetime
import logging
from typing import Union
from scipy.io import loadmat
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
import torch
import matplotlib.colors as mcolors
from scipy.ndimage import rotate
import glob
from AFNO import CTult_dataset2

# set flags / seeds
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FcnSingle(nn.Module):
    def __init__(self, planes: list or tuple, activation="gelu", last_activation=False):
        # =============================================================================
        #     Inspired by M. Raissi a, P. Perdikaris b,∗, G.E. Karniadakis.
        #     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
        #     involving nonlinear partial differential equations".
        #     Journal of Computational Physics.
        # =============================================================================
        super(FcnSingle, self).__init__()
        self.planes = planes
        if activation:
            self.active = nn.GELU()
        else:
            self.active = nn.Identity()
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))

        if last_activation:
            self.layers.append(self.active)
        self.layers = nn.Sequential(*self.layers)  # *的作用是解包

        # self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        out_var = self.layers(in_var)
        return out_var

class DeepONetMulti(nn.Module):
    def __init__(self, input_dim=2, operator_dims=[480 * 480 * 3], output_dim=2,
                 planes_branch=[128, 256, 512], planes_trunk=[128, 256, 512], activation='gelu',
                 learning_rate=1e-4,
                 step_size=100,
                 gamma=0.5,
                 weight_decay=1e-5,
                 eta_min=5e-4, grid=None):
        super(DeepONetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 1, 2, padding=1)
        self.up1 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.bnup = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(0.25)
        # self.branches = nn.ModuleList() # 分支网络
        self.trunks = nn.ModuleList()  # 主干网络
        self.branch = FcnSingle([3] + planes_branch, activation=activation)
        # for dim in operator_dims:
        #     self.branches.append(FcnSingle([dim] + planes_branch, activation=activation))# FcnSingle是从basic_layers里导入的
        for _ in range(output_dim):
            self.trunks.append(FcnSingle([input_dim] + planes_trunk, activation=activation))
        self.output_dim = output_dim
        self.reset_parameters()
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.grid = grid

        self.criterion = nn.MSELoss()
        self.criterion_val = nn.MSELoss()

    def reset_parameters(self):  # 初始化所有网络的参数
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, u_vars, y_var, size_set=True):
        # downsample
        # A = self.conv1(u_vars.unsqueeze(1))
        # A = nn.functional.gelu(A)
        # # A = nn.functional.max_pool2d(A, 2, 2)
        # # A = self.dropout(A)
        # A = self.conv2(A)
        # A = nn.functional.gelu(A)
        # # A = nn.functional.max_pool2d(A, 2, 2)
        # A = self.bn2(A)
        # # A = self.dropout(A)
        # # A = self.conv3(A)
        # # A = nn.functional.gelu(A)
        # # A = nn.functional.max_pool2d(A, 2, 2)
        # batch_size = A.shape[0]
        B = 1.
        # # print(u_vars.shape,y_var.shape)
        # # for u_var, branch in zip(u_vars, self.branches):
        # #     B *= branch(u_var)
        # B *= self.branch(A.reshape(batch_size,-1))

        batch_size = u_vars.shape[0]
        B *= self.branch(u_vars.reshape(batch_size, -1, 3))

        if not size_set:
            B_size = list(y_var.shape[1:-1])
            for i in range(len(B_size)):
                B = B.unsqueeze(1)
            B = torch.tile(B, [1, ] + B_size + [1, ])
        # print(B.shape,B.unsqueeze(1).shape)
        out_var = []
        for trunk in self.trunks:
            T = trunk(y_var)
            # print('T: ',T.shape)
            # print((B.unsqueeze(1) * T).shape)

            # upsampling
            # out = torch.sum(B.unsqueeze(1) * T, dim=-1).reshape(batch_size,1,120,120)
            out = torch.sum(B * T, dim=-1)

            # upsample
            # out = self.bnup(self.up1(out))
            # print(out.shape)
            # out = self.up2(out)
            # print(out.shape)
            
            out_var.append(out)
            # out_var.append(torch.sum(B.unsqueeze(1) * T, dim=-1)) # 用这种方式实现两个网络的乘积

        # print('out_var: ',out_var[0].shape)
        out_var = torch.stack(out_var, dim=-1)
        # print(out_var.shape)

        return out_var

    # def training_step(self, batch: torch.Tensor, batch_idx):
    #     # One step training
    #     field, speed, src = batch
    #     batch_size = speed.shape[0]
    #     grid = self.grid.view(1, -1, 2).repeat(batch_size, 1, 1).to(device)
    #     u_var = torch.cat([speed.unsqueeze(-1), src], dim=-1).to(device)
    #     field = field.to(device)
    #     # u_var = u_var.permute(0, 3, 1, 2).contiguous()
    #     # print(u_var.shape, grid.shape)
    #     out = self(u_var, grid)
    #     loss = self.criterion(out.reshape(batch_size, -1), field.reshape(batch_size, -1))
    #     self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    # def validation_step(self, val_batch: torch.Tensor, batch_idx):
    #     # One step validation
    #     field, speed, src = val_batch
    #     batch_size = speed.shape[0]
    #     grid = self.grid.view(1, -1, 2).repeat(batch_size, 1, 1).to(device)
    #     u_var = torch.cat([speed.unsqueeze(-1), src], dim=-1).to(device)
    #     field = field.to(device)
    #     # u_var = u_var.permute(0, 3, 1, 2).contiguous()
    #     out = self(u_var, grid)
    #     val_loss = self.criterion_val(out.reshape(batch_size, -1), field.reshape(batch_size, -1))
    #     self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
    #     return val_loss
    #
    # def configure_optimizers(self, optimizer=None, scheduler=None):
    #     if optimizer is None:
    #         optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    #     if scheduler is None:
    #         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.step_size, eta_min=self.eta_min)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler
    #         },
    #     }
if __name__ == '__main__':
    print("start")
    res = 480
    gridx = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridx = gridx.reshape(1, res, 1, 1).repeat([1, 1, res, 1])
    gridy = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, res, 1).repeat([1, res, 1, 1])
    grid = torch.cat((gridx, gridy), dim=-1).reshape(1, -1, 2)
    print(grid.shape)

    data_dir = "D:/AI4Sdata/data/data/dataset_train_1"
    start_i = 1
    end_i = 1
    is_transformed = True
    batch_size = 2
    homo_dir = 'D:\AI4Sdata\data\\u_homo.npy'
    train_dataset = CTult_dataset2(data_dir=data_dir, homo_dir=homo_dir, start_i=1, end_i=10,
                                  is_transformed=is_transformed)
    test_dataset = CTult_dataset2(data_dir=data_dir, homo_dir=homo_dir, start_i=200, end_i=200,
                                 is_transformed=is_transformed)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for field_pos, speed_pol,src in train_dataloader:
        print(field_pos.shape, speed_pol.shape,src.shape)
        break
    train_tag = True
    # train_tag = False
    Continue = True
    Continue = False
    if train_tag:
        model = DeepONetMulti(grid=grid).to(device)
        if Continue:
            model = torch.load('test_modelD.pt').to(device)
        mse_losses = []  # Initialize an empty list to store MSE losses
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        num_epoches = 10
        crit = nn.MSELoss()
        for epoch in range(num_epoches):
            total_loss = 0.0
            num_batches = 0
            tqdm_item = tqdm(data_loader)
            for field_pos, speed_pol, src in tqdm_item:
                input_data = torch.unsqueeze(speed_pol, dim=-1).type(torch.float32)
                input_data = input_data.to(device)
                y_true = field_pos

                y_true = y_true.to(device)
                src = src.to(device)
                y_pred = model(input_data, src)
                loss = crit(y_pred.reshape(batch_size, -1), y_true.reshape(batch_size, -1))
                mse_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate total loss
                total_loss += loss.item()
                loss_numeric = loss.item()
                num_batches += 1
                tqdm_item.set_description(f'Epoch[{epoch}/{num_epoches}]')
                tqdm_item.set_postfix(loss=loss.item())
                # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch + 1}/{num_epoches}], Average Loss: {avg_loss}')

        ### Print and plot the MSE loss
        plt.plot(mse_losses, marker='o')  # Plot the MSE losses
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.title('MSE Loss Over Iterations')
        # plt.grid(True)
        plt.show()
        torch.save(model, 'test_modelD.pt')

    for batch_idx, batch in enumerate(train_dataloader):
        field, speed, src = batch
        batch_size = speed.shape[0]
        u_var = torch.cat([speed.unsqueeze(-1), src], dim=-1)
        out = model.forward(u_var, grid)
        break
    fig = plt.figure(figsize=(80, 80))
    for i in range(4):
        plot_img = out[i, ..., 0]
        ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
        ax.imshow(plot_img, cmap="RdBu_r")
        ax.set_title(f'Out Id: {i}')
    fig.subplots_adjust(wspace=-0.8)
    plt.show()



