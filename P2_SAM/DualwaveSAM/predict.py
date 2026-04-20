import datetime

import numpy as np
import warnings
import pandas as pd
import glob

from medpy.metric.binary import hd,hd95
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import argparse

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ex_transforms

import metrics

from tqdm import *
warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')       # 128
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')   # 128

parser.add_argument('--input_nc', type=int, default=2, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')

parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')

opt = parser.parse_args()




def testOne(model,device,pid):


    if model == 'DualwaveSAM':
        net_g = torch.load('results/CHEN/TBME-DualwaveSAM/re_haar/best_model/net_model.pth', map_location=device)
    else:
        raise ValueError("input net error")

    
    Dice = []

    # test
    tal_dice = 0.0
    tal_hd95 = 0.0

    net_g.eval()
    inputs = []
    targets = []
    preds = []

    
    with torch.no_grad():
        num = 1
        #val_qbar
        val_qbar = trange(len(med_set_loader))
        
        for sample in med_set_loader:
            input, target = sample['input'].to(device), sample['target'].to(device)

            # target = torch.where(target == 2, 0, target)    # HECKTOR 只计算一个label

            # 获取 target 的唯一值
            unique_values = torch.unique(target)
            # 如果 target 的唯一值是 0，则将其添加到 preds 列表中
            if len(unique_values) == 1 and unique_values.item() == 0:
                preds.append(target.cpu().numpy())
                targets.append(target.cpu().numpy())
                num += 1
                tal_dice += 1.0
                tal_hd95 += 0.0
                val_qbar.set_postfix(dice=1.0,hd95=0.0)
                val_qbar.update(1)
                continue

            if model == 'DualwaveSAM':
                prediction,_ = net_g(input)
            else:
                prediction = net_g(input)
            dice_pred = metrics.dice(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
            try:
                hd = hd95(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
            except:
                hd = np.float64(0)
            tal_dice += dice_pred
            tal_hd95 += hd

            # inputs.append(input.cpu().numpy())
            preds.append(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()))
            targets.append(target.cpu().numpy())

            num += 1
            val_qbar.set_postfix(dice=dice_pred.item(),hd95=hd.item())  # 进度条右边显示信息
            val_qbar.update(1)

        # 计算平均指标
        avg_dice = tal_dice / len(med_set_loader)
        avg_hd95 = tal_hd95 / len(med_set_loader)

        Dice.append(avg_dice)

        file = os.path.join("predict_one/test33", pid)
        # 存储 test.npz
        np.savez(file, pred=preds, target=targets)  # 存储标签和预测

        print('*' * 100)
        print("test avg_dice: {:.4f} ".format(avg_dice))
        print("test avg_HD95: {:.4f}".format(avg_hd95))
        print('*' * 100)




class HECKTORdataset(Dataset):
    """
    按切片级别划分训练/测试集，缓存有效切片，避免重复计算。
    """
    # 类级缓存
    _slice_inputs = None
    _slice_targets = None

    def __init__(
        self,
        mode: str = 'test',
        seed: int = 17,
        split_rate: float = 0.8,
        data_rate: float = 1.0,
        transform=None,
    ):
        self.transform = transform
        self.mode = mode

        if 1:

            # (n, h, w, d, c)
            # (256, 256, 590, 2)
            # (256, 256, 590)

            # 调整维度: (H, W, D, C) -> (D, H, W, C)
            inputs_cases  = np.moveaxis(_all_inputs,  [2,0,1], [0,1,2])
            # (H, W, D) -> (D, H, W)
            targets_cases = np.moveaxis(_all_targets, [2,0,1], [0,1,2])

            # 收集所有切片
            HECKTORdataset._slice_inputs  = inputs_cases  # (num_slices, H, W, C)
            HECKTORdataset._slice_targets = targets_cases # (num_slices, H, W)

        # 获取缓存的有效切片
        inputs_slices  = HECKTORdataset._slice_inputs
        targets_slices = HECKTORdataset._slice_targets

        num_slices = inputs_slices.shape[0]

        # 切片数据
        self.inputs  = inputs_slices
        self.targets = targets_slices

        print(f"模式: {mode} | 总切片: {num_slices} ")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = dict()

        sample['id'] = idx  

        sample['input'] = self.inputs[idx]     
        sample['target'] = self.targets[idx]  

        sample['input'] = cv2.resize(sample['input'], (256, 256), interpolation=cv2.INTER_LINEAR)   # (256,256,2)
        sample['input'] = sample['input'].transpose([1, 0, 2])
        # print(sample['input'].shape)

        sample['target'] = cv2.resize(sample['target'], (256, 256), interpolation=cv2.INTER_LINEAR)
        sample['target'] = np.expand_dims(sample['target'], axis=2)                                 # (256,256,1)
        sample['target'] = sample['target'].transpose([1, 0, 2])
        # print(sample['target'].shape)

        if self.transform:
            sample = self.transform(sample)
        return sample
    



if __name__ == "__main__":

    curr_time = datetime.datetime.now()

    # 分配 gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = 'DualwaveSAM' 
    print('\n' + '*'*30 + f"测试模型{model_name}中" + '*'*30)

    # 获取 dataset/CHEN_test 文件夹下的所有病人
    npz_files = glob.glob('dataset/CHEN_test/*.npz')
    pid_list = [os.path.splitext(os.path.basename(file))[0] for file in npz_files] 
    print(pid_list)

    for pid in pid_list:
        print(f"\n开始处理 PID: {pid}")

        _npz_path = f"dataset/CHEN_test/{pid}.npz" 
        _dataset = np.load(_npz_path, allow_pickle=True)
        _all_inputs  = _dataset['input']   # (N_pat, H, W, D, C)
        _all_targets = _dataset['target']  # (N_pat, H, W, D)


        if opt.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(opt.seed)
        if opt.cuda:
            torch.cuda.manual_seed(opt.seed)


        med_transform = ex_transforms.Compose([
                                                ex_transforms.ToTensor()
                                            ])

        med_data = HECKTORdataset(mode='test',transform=med_transform,seed=42) 


        med_set_loader = DataLoader(dataset=med_data, num_workers=opt.threads, batch_size=opt.test_batch_size,
                                    shuffle=False, drop_last=True)

        testOne(model_name,device,pid)


    print('\n' + '*'*30 + f"模型{model_name}测试完成" + '*'*30)