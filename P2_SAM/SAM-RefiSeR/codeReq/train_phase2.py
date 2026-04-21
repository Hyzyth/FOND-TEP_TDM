import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import gc
import nibabel as nib
import tqdm as tqdm
from utils.meter import AverageMeter
from utils.general import save_checkpoint, load_pretrained_model, resume_training
from brats import get_domain_adaptation_datasets_alwayst2_singletarget, get_domain_adaptation_datasets_train_selective
from tqdm import tqdm
import datetime
from FDA import FDA_source_to_target_3d

from iterative_training_utils import (
    PseudoTargetDataset
)

from monai.data import  decollate_batch
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torch.utils.data.dataset import Dataset, ConcatDataset
import torch.nn.functional as F


from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.networks.nets import SwinUNETR, VNet, AttentionUnet, UNETR
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from networks.models.UNet.model import UNet3D, GRLUNet3D
from networks.models.UX_Net.network_backbone import UXNET
from networks.models.nnformer.nnFormer_tumor import nnFormer
from networks.models.SegResNet.segresnet import SegResNet, GRLSegResNet, GraphGRLSegResNet

from codes_bruv.svd_loss import svd_loss_

try:
    from thesis.models.SegUXNet.model import SegUXNet
except ModuleNotFoundError:
    print('model not available, please train with other models')
    
from functools import partial
from utils.augment import DataAugmenter
from utils.schedulers import SegResNetScheduler, PolyDecayScheduler

# Configure logger
import logging
import hydra
from omegaconf import DictConfig



###########################################

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

###########################################


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs("logger", exist_ok=True)
file_handler = logging.FileHandler(filename="logger/train_logger.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

torch.autograd.set_detect_anomaly(True)

def save_best_model(dir_name, model, name="best_model"):
    save_path = os.path.join(dir_name, name)
    os.makedirs(save_path, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict, f"{save_path}/{name}.pkl")

def setup_experiment_logging(exp_folder):
    log_file = os.path.join(exp_folder, "experiment_log.txt")

    with open(log_file, 'w') as f:
        f.write('experiment log\n')
    return log_file

def log_message(log_file, message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(f"{timestamp}: {message}\n")

def save_checkpoint(dir_name, state, name="checkpoint"):
    save_path = os.path.join(dir_name, name)
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, f"{name}.ptr.tar")
    torch.save(state, checkpoint_path)



def create_dirs(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "best-model"), exist_ok=True)


def init_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Solver:
    def __init__(self, model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-5):
        self.lr = lr
        self.weight_decay = weight_decay
        self.all_solvers = {
            "Adam": torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True),
            "AdamW": torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True),
            "SGD": torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay),
        }

    def select_solver(self, name):
        return self.all_solvers[name]





def dann_alpha_schedule(
    epoch: int,
    max_epochs: int,
    warmup: int = 1,
    initial: float = 0.0,       
    ceiling: float = 0.15,     
    steepness: float = 3.0,
    cap_epoch: int = 15         
) -> float:
    if epoch < warmup:
        return 0.0
    if epoch >= cap_epoch:
        return ceiling
    p = (epoch - warmup) / float(max_epochs - warmup)
    ramp = 1.0 / (1.0 + np.exp(-steepness * (p - 0.5)))
    return float(np.clip(initial + (ceiling - initial) * ramp,
                         initial, ceiling))


def lambda_domain_schedule(
    epoch: int,
    max_epochs: int,
    warmup: int = 1,
    initial: float = 0.05,
    ceiling: float = 0.08,    
    steepness: float = 3.0,
    cap_epoch: int = 15       
) -> float:
    if epoch < warmup:
        return 0.0
    if epoch >= cap_epoch:
        return ceiling
    p = (epoch - warmup) / float(max_epochs - warmup)
    ramp = 1.0 / (1.0 + np.exp(-steepness * (p - 0.5)))
    return float(np.clip(initial + (ceiling - initial) * ramp,
                         initial, ceiling))






def lambda_svd_schedule(current_epoch: int, max_epochs: int) -> float:
    p = float(current_epoch) / float(max_epochs)
    lambda_svd_val = 2.0 / (1.0 + np.exp(-0.01 * p)) - 1
    return lambda_svd_val

def apply_FDA_from_random_target(source_img, target_dataset, L = 0.425):
    random_idx = random.randint(0, len(target_dataset) - 1)
    target_sample = target_dataset[random_idx]
    target_img = target_sample['image']

    src_img_batch = source_img.unsqueeze(0)
    trg_img_batch = target_img.unsqueeze(0)

    modified_img_batch = FDA_source_to_target_3d(src_img_batch, trg_img_batch, L = L)

    modified_img = modified_img_batch.squeeze(0)
    return modified_img

def train_epoch_domain_adaptation(model,
                                  source_loader,
                                  target_loader,
                                  optimizer,
                                  device,
                                  epoch: int,
                                  max_epochs: int,
                                  cfg,
                                  schedule_alpha: bool = True,
                                  schedule_lambda: bool = True,
                                  schedule_svd: bool = True):
    model.train()
    epoch_loss_meter = AverageMeter()
    scaler = GradScaler('cuda')

    if schedule_alpha:
        alpha = dann_alpha_schedule(epoch, max_epochs)
    else:
        alpha = 0

    if schedule_lambda:
        lambda_domain = lambda_domain_schedule(epoch, max_epochs)
    else:
        lambda_domain = 0

    if schedule_svd:
        lambda_svd = lambda_svd_schedule(epoch, max_epochs)
    else:
        lambda_svd = 0

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    num_steps = max(len(source_loader), len(target_loader))

    total_domain_loss = 0.0
    total_graph_loss = 0.0
    total_seg_loss = 0.0

    real_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    disc_params = list(real_model.domain_discriminator.parameters())

    with tqdm(range(num_steps), desc=f"DomainAdapt-Train Epoch {epoch + 1}", leave=False) as progress_bar:
        for step_idx in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            try:
                batch_data_s = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                batch_data_s = next(source_iter)

            try:
                batch_data_t = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                batch_data_t = next(target_iter)

            images_s = batch_data_s['image'].to(device)  
            seg_labels_s = batch_data_s['label'].to(device)
            domain_labels_s = batch_data_s['domain_label'].to(device)
            images_t = batch_data_t['image'].to(device)  
            seg_labels_t = batch_data_t['label'].to(device)
            domain_labels_t = batch_data_t['domain_label'].to(device)

            if cfg.training.apply_fda:
                B = images_s.size(0)
                modified_src_list = []
                for i in range(B):
                    j = random.randint(0, images_t.size(0) - 1)
                    src_img = images_s[i].unsqueeze(0)  
                    trg_img = images_t[j].unsqueeze(0)    
                    mod_img = FDA_source_to_target_3d(src_img, trg_img, L=cfg.training.FDAL)
                    mask = (src_img > 0).float()
                    composite_img = mask * mod_img + (1 - mask) * src_img
                    modified_src_list.append(composite_img.squeeze(0))
                images_s = torch.stack(modified_src_list, dim=0)

            with autocast("cuda"):
                seg_loss_s, domain_loss_s, pooled_feat_s = model(
                    images_s,
                    seg_labels=seg_labels_s,
                    domain_labels=domain_labels_s,
                    alpha=alpha,
                    return_features=True
                )
                total_loss_s = float(cfg.training.lambda_source) * seg_loss_s + float(lambda_domain) * domain_loss_s
            total_seg_loss += seg_loss_s.item()

            with autocast("cuda"):
                seg_loss_t, domain_loss_t, pooled_feat_t = model(
                    images_t,
                    seg_labels=seg_labels_t,
                    domain_labels=domain_labels_t,
                    alpha=alpha,
                    return_features=True
                )
                total_loss_t = float(cfg.training.lambda_pseudo) * seg_loss_t + float(lambda_domain) * domain_loss_t

            if lambda_svd == 0:
                total_loss_svd = torch.zeros(1, device=device, dtype=torch.float, requires_grad=True)
                svd_loss = total_loss_svd
            else:
                with torch.autocast(device_type='cuda', enabled=False):
                    svd_loss = svd_loss_(pooled_feat_s.float(), pooled_feat_t.float())
                total_loss_svd = torch.tensor(lambda_svd, device=device, dtype=torch.float) * svd_loss

            total_loss_all = total_loss_s + total_loss_t + total_loss_svd
            
            
  

            scaler.scale(total_loss_all).backward()
            scaler.unscale_(optimizer)  
            torch.nn.utils.clip_grad_norm_(disc_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()



            total_domain_loss += (domain_loss_s.item() + domain_loss_t.item()) / 2
            total_graph_loss += svd_loss.item()

            total_loss_val = total_loss_s.item() + total_loss_t.item() + total_loss_svd.item()
            epoch_loss_meter.update(total_loss_val, images_s.size(0) + images_t.size(0))

            progress_bar.set_postfix({
                "alpha": f"{alpha:.3f}",
                "lambda_domain": f"{lambda_domain:.3f}",
                "lambda_svd": f"{lambda_svd:.3f}",
                "src_seg_loss": f"{seg_loss_s.item():.3f}",
                "src_dom_loss": f"{domain_loss_s.item():.3f}",
                "tgt_dom_loss": f"{domain_loss_t.item():.3f}",
                "svd_loss": f"{svd_loss.item():.3f}",
            })

    avg_domain_loss = total_domain_loss / num_steps
    avg_graph_loss = total_graph_loss / num_steps
    avg_seg_loss = total_seg_loss / num_steps

    return epoch_loss_meter.avg, avg_domain_loss, avg_graph_loss, avg_seg_loss, lambda_domain, lambda_svd, alpha



def val(model, loader, acc_func, device, model_inferer=None, post_sigmoid=None, post_pred=None):


    model.eval()
    run_acc = AverageMeter()

    with torch.no_grad():
        for batch_data in loader:
            
            logits = model_inferer(batch_data["image"].to(device))
            masks = decollate_batch(batch_data["label"].to(device))

            prediction_lists = decollate_batch(logits)
            predictions = [post_pred(post_sigmoid(pred)) for pred in prediction_lists]

            acc_func.reset()
            acc_func(y_pred=predictions, y=masks)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

    return run_acc.avg


def save_data(training_loss, et, wt, tc, val_mean_acc, epochs, cfg):
    data = {}
    NAMES = ["training_loss", "WT", "ET", "TC", "mean_dice", "epochs"]
    data_lists = [training_loss, wt, et, tc, val_mean_acc, epochs]
    for i in range(len(NAMES)):
        data[f"{NAMES[i]}"] = data_lists[i]
    data_df = pd.DataFrame(data)
    save_path = os.path.join(cfg.training.exp_name, "csv")
    os.makedirs(save_path, exist_ok=True)
    data_df.to_csv(os.path.join(save_path, "training_data.csv"))
    return data


def trainer(
    cfg,
    model,
    source_loader,
    target_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    device,
    model_inferer=None,
    max_epochs=100,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    val_every=10,
    log_file = None
    ):

    val_acc_max = 0
    dices_tc, dices_wt, dices_et, mean_dices = [], [], [], []
    epoch_losses, train_epochs = [], []
    best_train_loss = float('inf')

    

    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()

        train_loss, avg_domain_loss, avg_graph_loss, avg_seg_loss, lambda_domain, lambda_svd, alpha = train_epoch_domain_adaptation(
            model = model,
            source_loader = source_loader,
            target_loader = target_loader,
            optimizer = optimizer,
            device = device,
            epoch = epoch,
            max_epochs = max_epochs,
            cfg = cfg,
            schedule_alpha = cfg.training.schedule_alpha,
            schedule_lambda = cfg.training.schedule_lambda,
            schedule_svd = cfg.training.schedule_svg
        )

        epoch_time = (time.time() - epoch_start_time) / 60.0

        log_message(log_file,
            f"Epoch {epoch+1}/{max_epochs} TRAIN — "
            f"seg_loss={avg_seg_loss:.4f}, "
            f"domain_loss={avg_domain_loss:.4f}, "
            f"graph_loss={avg_graph_loss:.4f}, "
            f"lambda_domain={lambda_domain:.4f}, "
            f"alpha={alpha:.4f}, "
            f"time={epoch_time:.2f}min"
        )

        scheduler.step()


        if (epoch % val_every == 0) or (epoch == 0):
            epoch_losses.append(train_loss)
            train_epochs.append(epoch)

            val_acc = val(
                model=model,
                loader=val_loader,
                acc_func=acc_func,
                device=device,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_et = val_acc[0]
            dice_wt = val_acc[1]
            dice_tc = val_acc[2]
            mean_dice = np.mean(val_acc)

            dices_et.append(dice_et)
            dices_wt.append(dice_wt)
            dices_tc.append(dice_tc)
            mean_dices.append(mean_dice)

            log_message(log_file, 
                f"Epoch {epoch+1}/{max_epochs}, train_loss={train_loss:.4f}, seg_loss_s={avg_seg_loss}, domain_loss={avg_domain_loss:.4f}, graph_loss={avg_graph_loss:.4f}, lambda_domain={lambda_domain:.4f}, lambda_svd={lambda_svd:.4f}, alpha={alpha:.4f}"
                f" time={epoch_time:.2f} min, Val => ET={dice_et:.4f}, TC={dice_wt:.4f}, WT={dice_tc:.4f}, mean={mean_dice:.4f}")
        
            if cfg.training.checkpoint_criteria == 'train_loss':
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    save_best_model(cfg.training.exp_name, model, "best-model")
                    log_message(log_file, 'Best model (by training loss) is saved on this epoch')
            elif cfg.training.checkpoint_criteria == 'val':
                if mean_dice > val_acc_max:
                    val_acc_max = mean_dice
                    save_best_model(cfg.training.exp_name, model, "best-model")
                    log_message(log_file, 'Best model (by validation metric) is saved on this epoch')
                    save_checkpoint(
                        cfg.training.exp_name,
                        {
                            "epoch": epoch + 1,
                            "max_epochs": max_epochs,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        },
                        name = 'checkpoint',
                    )

                if epoch+1 == 96:
                    save_best_model(cfg.training.exp_name, model, "last-epoch-model")
                    log_message(log_file, 'Best model (by last-epoch) is saved on this epoch')

            save_best_model(cfg.training.exp_name, model, f"model-on-epoch-{epoch}")
            log_message(log_file, f"Model on epoch {epoch} saved")



        else:
            logger.info(
                f"Epoch {epoch+1}/{max_epochs}, train_loss={train_loss:.4f}, time={epoch_time:.2f} min"
            )

    logger.info(f"Training Finished! Best Validation Mean Dice: {val_acc_max:.4f}")
    save_data(
        training_loss=train_loss,
        et=dices_et,
        wt=dices_wt,
        tc=dices_tc,
        val_mean_acc=mean_dices,
        epochs=train_epochs,
        cfg=cfg
    )

    return val_acc_max


from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
    return new_state_dict


def run(
    cfg,
    model,
    device,
    loss_func,
    acc_func,
    optimizer,
    source_loader,
    target_loader,
    val_loader,
    scheduler,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
    max_epochs=100,
    val_every=10,
    log_file = None
):
    create_dirs(cfg.training.exp_name)

    start_epoch = 0
    if cfg.training.resume:
        logger.info("Resuming training...")
        pkl_path = os.path.join("/home/monetai/Desktop/dillan/allBrats/code/brrr/Brain-Tumors-Segmentation/archiveModels/t2_t1/t2_t1_phase6/best-model/best-model.pkl")
        raw_state = torch.load(pkl_path, map_location=device)

        state = OrderedDict()
        for k,v in raw_state.items():
            state[k.replace("module.","")] = v

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state)

        logger.info(f"Loaded checkpoint from {pkl_path}")


        start_epoch = 0
        max_epochs   = 150

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")


    best_val_mean = trainer(
        cfg=cfg,
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func = loss_func,
        acc_func=acc_func,
        scheduler=scheduler,
        device=device,
        model_inferer=model_inferer,
        max_epochs=max_epochs,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        val_every=val_every,
        log_file = log_file
    )
    return best_val_mean


@hydra.main(config_name="configs", config_path="conf", version_base=None)
def main(cfg: DictConfig):
    """Main function for distributed training."""
    exp_folder = cfg.training.exp_name
    os.makedirs(exp_folder, exist_ok = True)
    log_file = setup_experiment_logging(exp_folder)
    log_message(log_file, "Starting experiment.")

    init_random(seed=cfg.training.seed)

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True

    if cfg.model.architecture == "unet3d":
        model = UNet3D(in_channels = 1, num_classes = 3).to(device)
    elif cfg.model.architecture == "segres_net":
        model = SegResNet(spatial_dims=3, 
                    init_filters=32, 
                    in_channels=1, 
                    out_channels=3, 
                    dropout_prob=0.2, 
                    blocks_down=(1, 2, 2, 4), 
                    blocks_up=(1, 1, 1)).to(device)
    elif cfg.model.architecture == "grlsegres_net":
        model = GRLSegResNet(
            spatial_dims = 3,
            init_filters = 32,
            in_channels = 1,
            out_channels = 3,
            dropout_prob = 0.2,
            blocks_down = (1, 2, 2, 4),
            blocks_up = (1, 1, 1),
            num_domains = 2,
            alpha = 1.0
        ).to(device)
    elif cfg.model.architecture == 'grl_unet3d':
        model = GRLUNet3D(in_channels = 1, num_classes = 3, num_domains = 3,
                           level_channels=[64, 128, 256], bottleneck_channel=512).to(device)
    elif cfg.model.architecture == 'graph_grl_segresnet':
        model = GraphGRLSegResNet(
            spatial_dims = 3,
            init_filters = 32,
            in_channels = 1,
            out_channels = 3,
            dropout_prob = 0.2,
            blocks_down = (1, 2, 2, 4),
            blocks_up = (1, 1, 1),
            num_domains = 2,
            alpha = 1.0
        ).to(device)

    else:
        raise NotImplementedError("Please implement your chosen architecture init here...")


    base_model = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        base_model = model.module

    encoder_params = []
    disc_params    = []
    for name, param in base_model.named_parameters():
        if "domain_discriminator" in name:
            disc_params.append(param)
        else:
            encoder_params.append(param)

    base_lr = cfg.training.learning_rate
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': base_lr},
        {'params': disc_params,    'lr': base_lr * 0.1},
    ], weight_decay=cfg.training.weight_decay)


    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model._set_static_graph()


    dataset_dir = cfg.dataset.dataset_folder


    ds_source, ds_target = get_domain_adaptation_datasets_train_selective(
        dataset_folder = dataset_dir,
        mode = "train",
        target_size = (128, 128, 128),
        version = 'brats2020',
        seedAh = 42,
        chosen_modality_for_target = 't1',
        selected = 't1',
        fraction = 1
    )

    val_dataset = get_domain_adaptation_datasets_alwayst2_singletarget(
        dataset_folder=dataset_dir,
        mode="train_val",
        target_size=(128, 128, 128),
        version="brats2020",
        seedAh = 42,
        chosen_modality = 't1'
    )

    source_sampler = DistributedSampler(ds_source, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    target_sampler = DistributedSampler(ds_target, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
        if world_size > 1 else None

    source_loader = torch.utils.data.DataLoader(
        ds_source,
        batch_size=cfg.training.batch_size,
        shuffle=(source_sampler is None),
        sampler=source_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )

    target_loader = torch.utils.data.DataLoader(
        ds_target,
        batch_size=cfg.training.batch_size,
        shuffle=(target_sampler is None),
        sampler=target_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=False
    )

    loss_func = DiceLoss(to_onehot_y=False, sigmoid=True)
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)

    roi = cfg.model.roi
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi] * 3,
        sw_batch_size=cfg.training.sw_batch_size,
        predictor=model,
        overlap=cfg.model.infer_overlap
    )

    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    post_sigmoid = Activations(sigmoid=True)

    if rank == 0:
        logger.info(f"Starting DANN training with world_size={world_size}, local_rank={local_rank}")
        logger.info(f"Source dataset size={len(ds_source)},, "
                    f"Val dataset size={len(val_dataset)}")
        
    best_val_mean = run(
        cfg=cfg,
        model=model,
        device=device,
        loss_func = loss_func,
        acc_func=acc_func,
        optimizer=optimizer,
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        model_inferer=model_inferer,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        max_epochs=cfg.training.max_epochs,
        val_every=cfg.training.val_every,
        log_file = log_file
    )

    if world_size > 1:
        dist.destroy_process_group()


    if rank == 0:
        logger.info(f"Finished training. Best validation mean dice = {best_val_mean:.4f}")
        log_message(log_file, f"Finished training. Best Validation mean dice = {best_val_mean:.4f}")


if __name__ == "__main__":
    main()
