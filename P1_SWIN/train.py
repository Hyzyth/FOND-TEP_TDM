# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.inferers.utils import sliding_window_inference
from monai.losses.dice import DiceLoss, DiceCELoss
from monai.metrics.meandice import DiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms.post.array import AsDiscrete,Activations
from monai.transforms.compose import Compose
from networks.unetr import UNETR
from data_utils import get_loader # Modification: Change the reference to the .py in the root with the code changes. Original version stays in utils folder
from trainer import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
import argparse
from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import *
import warnings


#max époque, cache rate et warmum époque à spécifier en shell pour test soft

#CUDA_LAUNCH_BLOCKING=1 py -3.12 train.py --batch_size 2 --cache_rate 0.0 --max_epochs 2 --val_every 1 --workers 0 --logdir test_debug

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='for_log', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default=None, type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='Dataset_Final_SwinCross_SITK', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_swincross.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default=None, type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=3000, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=6, type=int, help='number of batch size')  # a reduire si besoin
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate') #valeur classique 
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')
parser.add_argument('--val_every', default=5, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--model_name', default='unetr', type=str, help='model name')
parser.add_argument('--pos_embed', default='perception', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--in_channels', default=2, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels') # MODIFICATION : the labels are 0 background, 1 tumour, 2 nodules
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--lower', default=0.0, type=float, help='a_min in ScaleIntensityRangePercentilesd')
parser.add_argument('--upper', default=100.0, type=float, help='a_max in ScaleIntensityRangePercentilesd')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRangePercentilesd')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRangePercentilesd')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.7, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--resume_jit', action='store_true', help='resume training from pretrained torchscript checkpoint')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')


parser.add_argument('--cache_rate', default=1.0, type=float, help='Ratio of dataset to cache in RAM (0.0=safe, 1.0=fast)')

def main():
    # GPU = [1]
    # gpus = ','.join([str(i) for i in GPU])
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = './runs/' + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        
        # --- DÉBUT DE LA MODIFICATION (REMPLACE LES ANCIENNES LIGNES) ---
    if torch.cuda.is_available():
        # Si un GPU est là, on le configure comme avant
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.benchmark = True
        print(f"✅ GPU détecté : Utilisation de cuda:{args.gpu}")
    else:
        # Sinon, on bascule tout sur CPU et on désactive les fonctions GPU-only
        args.device = torch.device("cpu")
        print("⚠️ Aucun GPU détecté (ou PyTorch CPU installé). Passage sur CPU.")
        args.distributed = False  # Impossible de faire du distribué sans GPU (souvent)
        args.amp = False          # L'AMP est optimisé pour GPU
    # --- FIN DE LA MODIFICATION ---

    args.test_mode = False
    loader = get_loader(args)

    # --- AJOUTE CECI JUSTE EN DESSOUS ---
    print("\n" + "="*30)
    print(f"DEBUG: Nombre de batchs dans le TRAIN loader : {len(loader[0])}")
    print(f"DEBUG: Nombre de batchs dans le VAL loader   : {len(loader[1])}")
    print("="*30 + "\n")
    # -----------------------------------

    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_x]
    pretrained_dir = args.pretrained_dir

    if (args.model_name is None) or args.model_name == 'unetr':
        config_sw = CONFIGS_sw_seg['SwinUNETR_CMFF-hecktor-v06']
        model = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)
        #model = SwinUNETR_dualModalityFusion_OutSum(config_sw)

    # MODIFICATION : Chargement des poids plus robuste (backbone fix)
        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name), map_location='cpu')
            # On essaie de charger tel quel d'abord, sinon on nettoie les clés
            try:
                model.load_state_dict(model_dict)
            except RuntimeError:
                print("Direct load failed, attempting to remove 'backbone.' prefix...")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_dict.items():
                    new_key = k.replace('backbone.', '') if 'backbone.' in k else k
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict, strict=False)
            print('Use pretrained weights')

        if args.resume_jit:
            if not args.noamp:
                print('Training from pre-trained checkpoint does not support AMP\nAMP is disabled.')
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))

    dice_loss = DiceCELoss(to_onehot_y=True,
                           softmax=True,
                           squared_pred=True,
                           smooth_nr=args.smooth_nr,
                           smooth_dr=args.smooth_dr)
    # NOUVEAU CODE (Compatible MONAI récent)
    # On passe directement le nombre de classes à to_onehot
    post_label = AsDiscrete(to_onehot=args.out_channels)
    
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=False,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=False)
    model_inferer = partial(sliding_window_inference,
                            roi_size=inf_size,
                            sw_batch_size=args.sw_batch_size,
                            predictor=model,
                            overlap=args.infer_overlap)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        model.load_state_dict(new_state_dict, strict=False)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.to(args.device)
    # Gestion du DDP (Distribué) seulement si GPU disponible
    if args.distributed and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(args.device) # Au cas où
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True)
    # ----------------------------------------------------------
    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(model=model,
                            train_loader=loader[0],
                            val_loader=loader[1],
                            optimizer=optimizer,
                            loss_func=dice_loss,
                            acc_func=dice_acc,
                            args=args,
                            model_inferer=model_inferer,
                            scheduler=scheduler,
                            start_epoch=start_epoch,
                            post_label=post_label,
                            post_pred=post_pred)
    return accuracy

if __name__ == '__main__':
    main()

