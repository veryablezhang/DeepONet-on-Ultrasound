# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fft as fft
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#from scipy.ndimage import rotate
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import cmath
import glob

Tensor = torch.Tensor

class AFNOMlp(nn.Module):
    """Fully-connected Multi-layer perception used inside AFNO

    Parameters
    ----------
    in_features : int
        Input feature size
    latent_features : int
        Latent feature size
    out_features : int
        Output feature size
    activation_fn :  nn.Module, optional
        Activation function, by default nn.GELU
    drop : float, optional
        Drop out rate, by default 0.0
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        activation_fn: nn.Module = nn.GELU(),
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, latent_features)
        self.act = activation_fn
        self.fc2 = nn.Linear(latent_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2DLayer(nn.Module):
    """AFNO spectral convolution layer

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    hidden_size_factor : int, optional
        Factor to increase spectral features by after weight multiplication, by default 1
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        if not (hidden_size % num_blocks == 0):
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape
        # Using ONNX friendly FFT functions
        x = fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_real, x_imag = torch.real(x), torch.imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(
            torch.einsum(
                "nyxbi,bio->nyxbo",
                x_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                x_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(
            torch.einsum(
                "nyxbi,bio->nyxbo",
                x_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                x_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[1],
            )
            + self.b1[1]
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 0
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 1
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        # TODO(akamenev): replace the following branching with
        # a one-liner, something like x.reshape(..., -1).squeeze(-1),
        # but this currently fails during ONNX export.
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        # Using ONNX friendly FFT functions
        x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    """AFNO block, spectral convolution and MLP

    Parameters
    ----------
    embed_dim : int
        Embedded feature dimensionality
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    mlp_ratio : float, optional
        Ratio of MLP latent variable size to input feature size, by default 4.0
    drop : float, optional
        Drop out rate in MLP, by default 0.0
    activation_fn: nn.Module, optional
        Activation function used in MLP, by default nn.GELU
    norm_layer : nn.Module, optional
        Normalization function, by default nn.LayerNorm
    double_skip : bool, optional
        Residual, by default True
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        activation_fn: nn.Module = nn.GELU(),
        norm_layer: nn.Module = nn.LayerNorm,
        double_skip: bool = True,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = AFNO2DLayer(
            embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        self.mlp = AFNOMlp(
            in_features=embed_dim,
            latent_features=mlp_latent_dim,
            out_features=embed_dim,
            activation_fn=activation_fn,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer

    Converts 2D patch into a 1D vector for input to AFNO

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
    ):
        super().__init__()
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        num_patches = (inp_shape[1] // patch_size[1]) * (inp_shape[0] // patch_size[0])
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.inp_shape[0]}*{self.inp_shape[1]})."
            )
        # (B,C,480,480)
        x = self.proj(x).flatten(2).transpose(1, 2)
        # (B,(480/patch_size)**2,embed_dim)
        return x


# @dataclass
# class MetaData:
#     name: str = "AFNO"
#     # Optimization
#     jit: bool = False  # ONNX Ops Conflict
#     cuda_graphs: bool = True
#     amp: bool = True
#     # Inference
#     onnx_cpu: bool = False  # No FFT op on CPU
#     onnx_gpu: bool = True
#     onnx_runtime: bool = True
#     # Physics informed
#     var_dim: int = 1
#     func_torch: bool = False
#     auto_grad: bool = False


class AFNO(torch.nn.Module):
    """Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    out_channels: int
        Number of output channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1

    Note
    ----
    Reference: Guibas, John, et al. "Adaptive fourier neural operators:
    Efficient token mixers for transformers." arXiv preprint arXiv:2111.13587 (2021).
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        out_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        if not (
            inp_shape[0] % patch_size[0] == 0 and inp_shape[1] % patch_size[1] == 0
        ):
            raise ValueError(
                f"input shape {inp_shape} should be divisible by patch_size {patch_size}"
            )

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            inp_shape=inp_shape,
            in_channels=self.in_chans,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )
        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass of core AFNO"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head(x)

        # Correct tensor shape back into [B, C, H, W]
        # [b h w (p1 p2 c_out)]
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        # [b h w p1 p2 c_out]
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        # [b c_out, h, p1, w, p2]
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])
        # [b c_out, (h*p1), (w*p2)]
        return out


def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

    return torch.cat((gridx, gridy), dim=-1).to(device)


class AFNO2D(nn.Module):
    def __init__(
            self,
            patch_size=[8, 8],
            embed_dim: int = 512,
            depth: int = 4,
            mlp_ratio: float = 4.0,
            drop_rate: float = 0.0,
            num_blocks: int = 16,
            sparsity_threshold: float = 0.01,
            hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()

        in_shape = [480, 480]
        in_channels = 6
        out_channels = 2
        self.afno = AFNO(
            inp_shape=in_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
        )

    def forward(self, input_data, src_data):
        """
        :param input_data: [B, 480, 480, 1] (speed)
        :param src_data: [B, 480, 480, 3] (angle + homo)
        :return: [B, 480, 480, 2]
        """

        grid = get_grid2D(input_data.shape, input_data.device)
        #print('grid:', grid.shape)
        # _input = torch.cat([input_data, src_data], dim=-1)  # [B, 480, 480, 4] ## remove grid
        # _input = input_data
        _input = torch.cat([input_data, grid, src_data], dim=-1)
        #print('_input:', _input.shape)
        #print('_input:', _input.shape)
        #print('input:', input_data.shape, 'grid:', grid.shape, 'scr:', src_data.shape)
        # field = src_data[..., 1:].clone()  # [B, 480, 480, 2]

        # input: [B, 480, 480, 6] to [B, 6, 480, 480]
        _input = _input.permute(0, 3, 1, 2).contiguous()

        # mag & phase
        # mag, phase = cmath.polar(_input)

        # afno
        pred = self.afno(_input)

        # output: [B, 6, 480, 480] to [B, 480, 480, 2]
        pred = pred.permute(0, 2, 3, 1).contiguous()

        return pred



class CTult_dataset2(Dataset):
    def __init__(self, data_dir, homo_dir, start_i=None, end_i=None, is_transformed=None):

        self.data_dir = data_dir
        self.start_i = start_i
        self.end_i = end_i
        self.is_transformed = is_transformed
        self.data_path_dict = dict()
        self.homo_dir = homo_dir

        data_folds = glob.glob(data_dir)
        for data_fold in data_folds:
            field_root = 'D:/AI4Sdata/data/field2'
            speed_root = os.path.join(data_fold, 'speed')

            field_files_list = glob.glob(field_root + '/*')
            speed_files_list = glob.glob(speed_root + '/*')
            max_point_files = glob.glob('D:/AI4Sdata/data/max_point/*')
            print(
                f"find {len(field_files_list)} field files and {len(speed_files_list)} speed files in {data_fold}")
            for speed_file_path in speed_files_list:
                sample_id = int(os.path.basename(speed_file_path).split('.')[0].split('_')[1])
                if sample_id < start_i or sample_id > end_i:
                    continue
                self.data_path_dict[sample_id] = {
                    'field_path': [os.path.join(field_root, f'train_{sample_id}_{i}.npy') for i in range(1, 5)],
                    'speed_path': speed_file_path,
                    'max_path': os.path.join('D:/AI4Sdata/data/max_point/', f'field_max_train_{sample_id}.npy')
                }

        print(f"find {len(max_point_files)} max_point files")
        print(f'find {len(self.data_path_dict)} samples')

    #### Step 1
    def __getitem__(self, index):
        get32 = True
        # get32 = False
        index2 = index
        if get32:
            index2 = index//32

        field_data = list()
        for sub_file in self.data_path_dict[self.start_i + index2]['field_path']:
            field_data.append(np.load(sub_file, mmap_mode='r'))
        field_pos = np.concatenate(field_data, axis=0)

        max_points = np.load(self.data_path_dict[self.start_i + index2]['max_path'], mmap_mode='r')
        homo = np.load(self.homo_dir)
        src = torch.from_numpy(homo)
        src_com = torch.complex(src[..., 0], src[..., 1])  # ampl and phase
        src_com /= 9570.81
        src_com = torch.cat(
            (torch.unsqueeze(torch.abs(src_com), dim=-1), torch.unsqueeze(torch.angle(src_com)/np.pi, dim=-1)), dim=-1)

        if not get32:
            # normalization
            for i in range(field_pos.shape[0]):
                x, y = max_points[i]
                field_max = np.abs(field_pos[i, int(x), int(y)])
                field_pos[i, :, :] /= field_max
                # print(field_max)

            # field_pos = np.concatenate(
            #     (np.real(field_pos)[:, :, :, np.newaxis], np.imag(field_pos)[:, :, :, np.newaxis]), axis=-1)
            field_pos = np.concatenate(
                (np.abs(field_pos)[:, :, :, np.newaxis], (np.angle(field_pos)/np.pi)[:, :, :, np.newaxis]), axis=-1)
        else:
            i = index%32
            x, y = max_points[i]
            field_max = np.abs(field_pos[i, int(x), int(y)])
            field_pos = field_pos[i,...]
            field_pos /= field_max
            theta = torch.tensor((i / 32 * 2 * np.pi) * np.ones((480, 480, 1),dtype='float32'))
            src_com = src_com[i,...]
            src_com = torch.cat((theta,src_com),-1)
            field_pos = np.concatenate(
                (np.abs(field_pos)[:, :, np.newaxis], (np.angle(field_pos)/np.pi)[:, :, np.newaxis]), axis=-1)

        speed_pol = np.load(self.data_path_dict[self.start_i + index2]['speed_path'])
        max = np.max(speed_pol)
        min = np.min(speed_pol)
        speed_pol = (speed_pol - min) / (max - min)
        # max = np.copy(max_points.astype(np.float32))

        if self.is_transformed:
            speed_rotated = self.speed_transform(speed_pol)
            return torch.from_numpy(field_pos), torch.from_numpy(speed_rotated)#, torch.from_numpy(max)

        # return torch.from_numpy(field_max_points), torch.from_numpy(field_cartpos), torch.from_numpy(field_pol_pos), torch.from_numpy(speed_rotated[index % 4 * 8:index % 4 * 8 + 8])
        return torch.from_numpy(field_pos), torch.from_numpy(speed_pol),src_com

    def __len__(self):
        return len(self.data_path_dict)*32

    def speed_transform(self, speed):

        rotated_speed = np.zeros([32, 480, 480], dtype='float32')
        for i in range(32):
            rotated = rotate(speed, angle=-i * 360 / 32, reshape=False, mode="nearest")
            rotated_speed[i, :, :] = rotated

        return rotated_speed

            ###DataLoader

import math
def criterion(y_pred, y_true_real, y_true_imag):
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must be the same")

    #mag_true, phase_true = cmath.polar(y_true)

    #y_true_real = torch.real(y_true)
    #y_true_imag  = torch.imag(y_true)
    mag_true = torch.sqrt((y_true_real)** 2 + (y_true_imag** 2)).flatten(start_dim=2)
    phase_true = torch.atan2(y_true_imag, y_true_real).flatten(start_dim=2)
    mse_mag = torch.mean((mag_true - y_pred[:, :, :, 0]) ** 2) # Calculate squared errors and take the mean
    mse_phase = torch.mean((phase_true - y_pred[:, :, :, 1]) ** 2)  # Calculate squared errors and take the mean
    mse   = mse_mag + mse_phase
    return mse

def field_vis(mag, phase, save_path):

    os.makedirs(save_path, exist_ok=True)
    #for i in range(data.shape[0]):
    mag = mag.cpu()
    phase = phase.cpu()

    # Visualize magnitude
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mag, cmap='gray')
    plt.colorbar()
    plt.title('Magnitude')

    # Visualize phase
    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='hsv')  # Using 'hsv' colormap for phase
    plt.colorbar()
    plt.title('Phase')

    plt.savefig(f"{save_path}.png")
    plt.show()
    plt.close()



if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is here")
else:
    device = torch.device('cpu')
    print("CUDA is not available, falling back to CPU...")


if __name__ == '__main__':
    print("start")

    data_dir = "D:/AI4Sdata/data/data/dataset_train_1"
    start_i = 1
    end_i = 3600
    is_transformed = None
    batch_size = 4
    homo_dir = 'D:\AI4Sdata\data\\u_homo.npy'
    dataset = CTult_dataset2(data_dir=data_dir,homo_dir = homo_dir, start_i=start_i, end_i=end_i, is_transformed=is_transformed)
    #     # data_loader = DataLoader(dataset, batch_size=7, shuffle=True, num_workers= 4)
    start_i_t = 3800
    end_i_t = 3810
    Test_dataset = CTult_dataset2(data_dir=data_dir, homo_dir=homo_dir, start_i=start_i_t, end_i=end_i_t,
                             is_transformed=is_transformed)
    Test_dataloader = DataLoader(Test_dataset, batch_size=32, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for field_pos, speed_pol,src in data_loader:
        print(field_pos.shape, speed_pol.shape,src.shape)
        break
    train_tag = True
    # train_tag = False
    Continue = True
    # Continue = False
    if train_tag:
        model = AFNO2D().to(device)
        if Continue:
            model = torch.load('test_model_p.pt').to(device)
        mse_losses = []  # Initialize an empty list to store MSE losses
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

        num_epoches = 10
        crit = nn.MSELoss()
        for epoch in range(num_epoches):
            total_loss = 0.0
            num_batches = 0
            tqdm_item = tqdm(data_loader)
            for field_pos, speed_pol, src in tqdm_item:
                input_data = torch.unsqueeze(speed_pol, dim=-1).type(torch.float32)
                # input_data = torch.flatten(speed_pol, 0, 1)
                # input_data = torch.unsqueeze(speed_pol, dim=1).type(torch.float32)
                # input_data = input_data.repeat(1,32,1,1)
                # input_data = torch.flatten(input_data, 0, 1)
                # input_data = torch.unsqueeze(input_data, dim=3)
                input_data = input_data.to(device)
                # print('input: ', input_data.shape)
                # y_true = torch.flatten(field_pos, 0, 1)
                y_true = field_pos

                #src_ri_split = torch.cat([torch.real(src_data), torch.imag(src_data)], dim=-1).type(torch.float32)
                y_true = y_true.to(device)
                # print('y_true: ',y_true.shape)

                # y_true_real = torch.real(y_true)
                # y_true_imag = torch.imag(y_true)
                #y_true = torch.view_as_complex(y_true)
                src = src.to(device)
                y_pred = model(input_data,src)
                #print('y_pred:', y_pred.shape, 'y_true_real:', y_true_real.shape, 'y_true_imag:', y_true_imag.shape)
                # print('y_pred:', y_pred.shape)
                # print('y_true:', y_true.shape)
                      # print(y_true[0,0,0,0])
                # loss = criterion(y_pred, y_true_real, y_true_imag)
                loss = crit(y_pred.reshape(batch_size,-1), y_true.reshape(batch_size,-1))
                # Append the MSE loss to the list
                mse_losses.append(loss.item()) # Convert the tensor to a scalar and append it

                # Backward pass and optimization
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
        plt.plot(mse_losses, marker='o') # Plot the MSE losses
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.title('MSE Loss Over Iterations')
        #plt.grid(True)
        plt.show()
        torch.save(model,'test_model_p.pt')
    else:
        model = torch.load('test_model_p.pt').to(device)

    for field_pos, speed_pol,src in data_loader:
        input_data = torch.unsqueeze(speed_pol, dim=-1).type(torch.float32)
        # input_data = input_data.repeat(1, 32, 1, 1)
        # input_data = torch.flatten(input_data, 0, 1)
        # input_data = torch.unsqueeze(input_data, dim=3)

        input_data = input_data.to(device)
        src = src.to(device)
        pred = model(input_data,src).cpu().detach().numpy()
        print(input_data.shape,src.shape,field_pos.shape, pred.shape)
        fig = plt.figure(figsize=(80, 80))
        plot_img = speed_pol[0, ...]
        # plot_img = src[1, ..., 2].cpu().detach().numpy()
        plot_img = pred[0, ..., 0]
        # plot_img = field_pos[0, ...,0]
        # plot_img = torch.real(field_pos[0, 0, ...])
        ax = fig.add_subplot(2, 1, 1, xticks=[], yticks=[])
        ax.imshow(plot_img, cmap="RdBu_r")
        # for i in range(32):
        #     plot_img = field_pos[0, i, ..., 1]
        #     ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
        #     ax.imshow(plot_img, cmap="RdBu_r")
        #     ax.set_title(f'Filed Id: {i}')
        plot_img = src[1, ...,1].cpu().detach().numpy() #ampl
        # plot_img = pred[0, ..., 0]
        # plot_img = input_data[33, ...].cpu().detach().numpy()
        plot_img = field_pos[0,...,1]
        ax = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
        ax.imshow(plot_img, cmap="RdBu_r")
        fig.subplots_adjust(wspace=-0.8)
        plt.show()
        # fig = plt.figure(figsize=(40, 40))
        # for i in range(32):
        #     plot_img = pred[i, ..., 1]
        #     ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
        #     ax.imshow(plot_img, cmap="RdBu_r")
        #     ax.set_title(f'Filed Id: {i}')
        # fig.subplots_adjust(wspace=-0.8)
        # plt.show()
        break
    for field_pos, speed_pol,src in Test_dataloader:
        input_data = torch.unsqueeze(speed_pol, dim=-1).type(torch.float32)
        input_data = input_data.to(device)
        src = src.to(device)
        pred = model(input_data,src).cpu().detach().numpy()
        for i in range(32):
            fig = plt.figure(figsize=(16, 16))
            plot_img = pred[i, ..., 0]
            ax = fig.add_subplot(2, 2, 1, xticks=[], yticks=[])
            ax.set_title('Pred Amplitude')
            ax.imshow(plot_img, cmap="RdBu_r")
            plot_img = field_pos[i, ..., 0]
            ax1 = fig.add_subplot(2, 2, 2, xticks=[], yticks=[])
            ax1.set_title('True Amplitude')
            ax1.imshow(plot_img, cmap="RdBu_r")
            plot_img = pred[i, ..., 1]
            ax2 = fig.add_subplot(2, 2, 3, xticks=[], yticks=[])
            ax2.set_title('Pred Phase')
            ax2.imshow(plot_img, cmap="RdBu_r")
            plot_img = field_pos[i, ..., 1]
            ax3 = fig.add_subplot(2, 2, 4, xticks=[], yticks=[])
            ax3.set_title('True Phase')
            ax3.imshow(plot_img, cmap="RdBu_r")
            # plt.show()
            # plt.savefig(f'''data/save/out{start_i_t}_{i}''')
        break

