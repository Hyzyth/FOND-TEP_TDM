import datetime
import numpy as np
import warnings
import pandas as pd

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


import losses
import metrics
from loguru import logger
from tqdm import *
warnings.filterwarnings("ignore")


from torch.optim import lr_scheduler
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=1.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)





# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=12, help='training batch size')     
parser.add_argument('--test_batch_size', type=int, default=12, help='testing batch size') 

parser.add_argument('--input_nc', type=int, default=2, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels') 

parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')    # epoch first
parser.add_argument('--niter_decay', type=int, default=40, help='# of iter to linearly decay learning rate to zero')   # epoch second

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam') 
# default='lambda'  ==》 scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')

parser.add_argument('--lr_decay_iters', type=int, default=1, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')
parser.add_argument('--lamb1', type=int, default=0.01, help='weight on L1 term in objective') #0.01
parser.add_argument('--lamb2', type=int, default=0.1, help='weight on L2 term in objective')  #0.1
parser.add_argument('--glr', type=int, default=0.000128, help='initial learning rate for SGD')
parser.add_argument('--trc', type=int, default=1, help='train count')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    logger.info("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)





def testOne(model,device):
    save_model = os.path.join(save_result_folder, mname+"-"+model) 
    if not os.path.exists(save_model):
        os.mkdir(save_model)
    #######  #######
    if model ==  'DualwaveSAM':
        # wavelet: haar/db2/sym4/bior4.4    
        # nowave
        # havede    
        # nolap
        fold = 'haar'   
    else:
        fold = 0
    #######  #######
    save_folder = os.path.join(save_model, 're_' + str(fold))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    logger.add(f"log/{CHOOSE_DATA}/{mname}-{model}-{fold}.log")


    if model == 'DualwaveSAM':
        from sam_wave import DualwaveSAM
        net_g = DualwaveSAM().to(device)
    else:
        raise ValueError("input net error")
    


    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    net_g_scheduler = get_scheduler(optimizer_g, opt)

    ganIterations = 0
    best_dice = 0
    best_epoch = 0
    Dice = []
    TDice = []
    Loss = []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        # train
        net_g.train()
        train_dice = 0
        train_loss = 0

        if model == 'DualwaveSAM':
            net_g.model.set_epoch(epoch)


        for i in range(opt.trc):
            #qbar
            qbar = trange(len(train_set_loader))

            for iteration, batch in enumerate(train_set_loader, 1):
                optimizer_g.zero_grad()
                # forward
                input, target = batch['input'].to(device), batch['target'].to(device)

                # target = torch.where(target == 2, 0, target)  # HECKTOR 只计算一个label

                if model == 'DualwaveSAM':
                    prediction, aux_pseudo_logits = net_g(input)
                else:
                    prediction = net_g(input)

                # print(input.shape)              # [64, 2, 256, 256]
                # print(prediction.shape)         # [64, 1, 256, 256]

                # loss function
                loss_g_l1 = criterionL1(prediction, target) * opt.lamb1
                loss_g_l2 = criterionMSE(prediction, target) * opt.lamb2

                if model == 'DualwaveSAM':
                    if aux_pseudo_logits != None:  
                        loss_g = 0.8*losses.Dice_and_FocalLoss()(prediction, target) + 0.2*losses.Dice_and_FocalLoss()(aux_pseudo_logits, target) + loss_g_l1 + loss_g_l2
                    else:
                        loss_g = losses.Dice_and_FocalLoss()(prediction, target) + loss_g_l1 + loss_g_l2
                else:
                    loss_g = losses.Dice_and_FocalLoss()(prediction, target) + loss_g_l1 + loss_g_l2


                loss_g.backward()
                optimizer_g.step()

                dice = metrics.dice(ex_transforms.ProbsToLabels()(prediction.detach().cpu().numpy()), target.detach().cpu().numpy())

                train_dice = train_dice + dice
                train_loss = train_loss + loss_g.item()

                qbar.set_postfix(epoch = epoch,loss=loss_g.item(),dice=dice.item())  # 进度条右边显示信息

                qbar.update(1)


        update_learning_rate(net_g_scheduler, optimizer_g)


        # test
        tal_dice = 0.0
        tal_hd95 = 0.0
        tal_voe = 0.0
        tal_rvd = 0.0
        tal_iou = 0.0
        tal_f1 = 0.0

        net_g.eval()
        inputs = []
        targets = []
        preds = []

        pred_save = os.path.join(save_folder, 'pred')
        if not os.path.exists(pred_save):
            os.mkdir(pred_save)
        pred_save_folder = os.path.join(pred_save, str(epoch))
        if not os.path.exists(pred_save_folder):
            os.mkdir(pred_save_folder)
        
        with torch.no_grad():
            num = 1
            #val_qbar
            val_qbar = trange(len(val_set_loader))
            
            for sample in val_set_loader:
                input, target = sample['input'].to(device), sample['target'].to(device)

                # target = torch.where(target == 2, 0, target)  # HECKTOR 只计算一个label

                if model == 'DualwaveSAM':
                    prediction, _ = net_g(input)
                else:
                    prediction = net_g(input)

                inputs.append(input.cpu().numpy())
                preds.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())

                dice_pred = metrics.dice(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                try:
                    hd = hd95(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                except:
                    hd = np.float64(0)
                tal_dice += dice_pred
                tal_hd95 += hd

                iou_pred = metrics.iou(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                tal_iou += iou_pred

                voe_pred = metrics.voe(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                tal_voe += voe_pred

                rvd_pred = metrics.rvd(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                tal_rvd += rvd_pred

                tal_f1s = [] 
                np_pred = ex_transforms.ProbsToLabels()(prediction.cpu().numpy())
                np_target = target.cpu().numpy() # (B, H, W)
                batch_size = np_pred.shape[0]
                for i in range(batch_size):
                    single_pred = np_pred[i]
                    single_target = np_target[i]
                    p, r, f1 = metrics.compute_lesion_metrics(single_pred, single_target)
                    tal_f1s.append(f1)
                tal_f1 += np.mean(tal_f1s)

                num += 1
                val_qbar.set_postfix(epoch = epoch,dice=dice_pred.item(),hd95=hd.item())  
                val_qbar.update(1)

            # 计算平均指标
            avg_dice = tal_dice / len(val_set_loader)
            avg_hd95 = tal_hd95 / len(val_set_loader)
            avg_voe = tal_voe / len(val_set_loader)
            avg_rvd = tal_rvd / len(val_set_loader)
            avg_iou = tal_iou / len(val_set_loader)
            avg_f1 = tal_f1 / len(val_set_loader)

            Dice.append(avg_dice)
            Loss.append(train_loss / (len(train_set_loader)*opt.trc))
            TDice.append(train_dice / (len(train_set_loader)*opt.trc))
            logger.info('*' * 100)
            logger.info(f"Epoch {epoch} metrics:")
            logger.info("train avg_loss: {:.4f} ".format(train_loss / (len(train_set_loader)*opt.trc)))
            logger.info("train avg_dice: {:.4f} ".format(train_dice / (len(train_set_loader)*opt.trc)))
            logger.info("test avg_dice: {:.4f} ".format(avg_dice))
            logger.info("test avg_HD95: {:.4f}".format(avg_hd95))
            logger.info("test avg_IoU: {:.4f}".format(avg_iou))
            logger.info("test avg_VOE: {:.4f}".format(avg_voe))
            logger.info("test avg_RVD: {:.4f}".format(avg_rvd))
            logger.info("test avg_F1: {:.4f}".format(avg_f1))
            

            # checkpoint
            if best_dice < avg_dice:
                best_dice = avg_dice
                best_epoch = epoch
                model_save = os.path.join(save_folder, 'best_model')
                if not os.path.exists(model_save):
                    os.mkdir(model_save)
                net_g_model_out_path = os.path.join(model_save, 'net_model.pth')
                torch.save(net_g, net_g_model_out_path)
                logger.info("Checkpoint saved to {}".format("checkpoint"))

                filenameg = model_save + '/' + "generator.tar"
                save_checkpoint(net_g, optimizer_g, filename=filenameg)
                a = []
                b = []
                a.append(best_dice)
                b.append(best_epoch)
                summary = pd.DataFrame({
                    'best_dice': a,
                    'best_epoch': b
                })
                file = os.path.join(save_folder, 'summary.csv')
                summary.to_csv(file, index=False, sep=';')
                file = os.path.join(save_folder, 'res')
                # 存储 res.npz
                np.savez(file, input = inputs, pred=preds, target=targets)


            logger.info("best dice: {:.4f} \t best epoch: {}".format(best_dice, best_epoch))
            logger.info('*' * 100)

    epoch = range(0, opt.niter + opt.niter_decay)   
    curr_time_1 = datetime.datetime.now()
    logger.info(f"计算时间:{curr_time_1 - curr_time}")


import os
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset

class HECKTORdataset(Dataset):
    """
    按病人级别预划分的目录加载切片数据，提取有效切片并缓存，避免重复计算。
    """
    # 类级缓存，改为字典按 mode 存储，避免 train 和 test 数据混淆
    _cache = {'train': None, 'test': None}

    def __init__(
        self,
        base_dir: str,       # 指定存放 train/test 文件夹的父级目录
        mode: str = 'train', # 'train' 或 'test'
        transform=None,
    ):
        self.transform = transform
        self.mode = mode

        # 如果当前模式的数据还没有被加载和缓存，则进行遍历读取
        if HECKTORdataset._cache[mode] is None:
            
            folder_path = base_dir + '_' + mode # CHEN_train/CHEN_test
            npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
            
            if len(npz_files) == 0:
                raise FileNotFoundError(f"未能在 {folder_path} 中找到任何 .npz 文件！")

            all_inputs_list = []
            all_targets_list = []

            # 遍历该模式下的每一个病人文件
            for fpath in npz_files:
                data = np.load(fpath, allow_pickle=True)
                patient_input = data['input']    # (H, W, D, C)
                patient_target = data['target']  # (H, W, D)

                if patient_input.ndim == 5 and patient_input.shape[0] == 1:
                    patient_input = patient_input[0]
                    patient_target = patient_target[0]

                # 调整维度: (H, W, D, C) -> (D, H, W, C)
                # 调整维度: (H, W, D)    -> (D, H, W)
                patient_input = np.transpose(patient_input, (2, 0, 1, 3))
                patient_target = np.transpose(patient_target, (2, 0, 1))

                # 找到该病人所有含肿瘤的切片索引
                # 1. mask值大于0
                has_any = patient_target.sum(axis=(1, 2)) > 0       # (D,)
                # 2. 该切片上至少有一个像素等于 1
                has_one = np.any(patient_target == 1, axis=(1, 2))  # (D,)
                # 3. 合并两个条件
                valid_mask = has_any & has_one                      # (D,)

                # 收集该病人的有效切片
                valid_inputs = patient_input[valid_mask]            # (num_valid_slices, H, W, C)
                valid_targets = patient_target[valid_mask]          # (num_valid_slices, H, W)

                all_inputs_list.append(valid_inputs)
                all_targets_list.append(valid_targets)

            # 将该模式下所有病人的有效切片合并并缓存
            HECKTORdataset._cache[mode] = {
                'inputs': np.concatenate(all_inputs_list, axis=0),
                'targets': np.concatenate(all_targets_list, axis=0)
            }

        # 获取缓存的切片
        inputs_slices = HECKTORdataset._cache[mode]['inputs']
        targets_slices = HECKTORdataset._cache[mode]['targets']
        num_slices = inputs_slices.shape[0]

        self.inputs = inputs_slices
        self.targets = targets_slices

        print(f"模式: {mode} | 包含病人文件数: {len(npz_files)} | 有效总切片: {num_slices} | 实际使用切片: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = dict()

        sample['id'] = idx  

        sample['input'] = self.inputs[idx]    
        sample['target'] = self.targets[idx]  

        sample['input'] = cv2.resize(sample['input'], (256, 256), interpolation=cv2.INTER_LINEAR)   
        sample['input'] = sample['input'].transpose([1, 0, 2])

        sample['target'] = cv2.resize(sample['target'], (256, 256), interpolation=cv2.INTER_NEAREST)
        sample['target'] = np.expand_dims(sample['target'], axis=2)                                 
        sample['target'] = sample['target'].transpose([1, 0, 2])

        if self.transform:
            sample = self.transform(sample)
            
        return sample







if __name__ == "__main__":

    curr_time = datetime.datetime.now()

    # 分配 gpu
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


    CHOOSE_DATA = "Hecktor"   # CHEN / Hecktor
    save_result_folder = f'results/{CHOOSE_DATA}'

    # ====== dataset 父目录 ======
    base_data_dir = f"/data/code/MMCA-Net/dataset/{CHOOSE_DATA}"


    mname = 'TBME'
    test_info = "00"


    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)


    mr_transform = ex_transforms.Compose([
                                            ex_transforms.Horizontal_Mirroring(p=0.5),
                                            ex_transforms.Vertical_Mirroring(p=0.5),
                                            ex_transforms.ToTensor()
                                            ])

    val_transform = ex_transforms.Compose([
                                            ex_transforms.ToTensor()
                                        ])

    # 初始化 Dataset
    train_data = HECKTORdataset(base_dir=base_data_dir, mode='train', transform=mr_transform) 
    val_data = HECKTORdataset(base_dir=base_data_dir, mode='test', transform=val_transform)


    print("\n")
    print(f"训练集切片数：{len(train_data)}")
    print(f"验证集切片数：{len(val_data)}")


    train_set_loader = DataLoader(dataset=train_data, num_workers=opt.threads, batch_size=opt.batch_size,
                                    shuffle=True, drop_last=True)
    val_set_loader = DataLoader(dataset=val_data, num_workers=opt.threads, batch_size=opt.test_batch_size,
                                shuffle=False, drop_last=True)



    logger.info(f"{len(train_set_loader)}, {len(val_set_loader)}")


    logger.info('===> Building models')


    ## ours
    nList = ['DualwaveSAM']


    for i in nList:
        print('\n' + '*'*30 + f"训练模型{i}中" + '*'*30)
        testOne(i,device)
        print('\n' + '*'*30 + f"模型{i}训练完成" + '*'*30)
