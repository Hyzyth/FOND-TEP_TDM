import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from utils.general import load_pretrained_model
from utils.all_utils import save_seg_csv, cal_confuse, cal_dice
from brats import get_datasets_prime
from utils.meter import AverageMeter
from scipy.ndimage import label, find_objects

import torch.multiprocessing as mp


from monai.metrics import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.networks.nets import SwinUNETR, VNet, BasicUNetPlusPlus, AttentionUnet, DynUNet, UNETR
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from networks.models.UNet.model import UNet3D
from networks.models.UX_Net.network_backbone import UXNET
from networks.models.nnformer.nnFormer_tumor import nnFormer
from networks.models.SegResNet.segresnet import SegResNet, GRLSegResNet, GraphGRLSegResNet

import shutil

try:
    from thesis.models.SegUXNet.model import SegUXNet
    from thesis.models.v2.model import SegSCNet
    from thesis.models.v3.model import SCFENet
except ModuleNotFoundError:
    print('model not available, please train with other models')
    # sys.exit(1)

from functools import partial
import random

import hydra
from omegaconf import OmegaConf, DictConfig
import logging
import os
from tqdm import tqdm
import glob
import pickle

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
os.makedirs("logger", exist_ok=True)
file_handler = logging.FileHandler(filename="logger/logger_test.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_value(value):
    """proprecess value to scaler"""
    if torch.is_tensor(value):
        return value.item()
    return value

def sliding_inference(model, input, batch_size, overlap):
    return sliding_window_inference(inputs = input, roi_size = (128, 128, 128),
                                    sw_batch_size = batch_size, predictor = model, overlap = overlap)

def inference(model, input, batch_size, overlap):
    return sliding_inference(model, input, batch_size, overlap)


def compute_bounding_box(mask, padding=10, debug=False):

    label_names = ['ET', 'TC', 'WT']
    boxes = {}
    for i, label_name in enumerate(label_names):
        m = mask[i]
        
        if debug:
            print(f"\nProcessing label '{label_name}' (channel index {i}):")
            print(f"  Mask shape: {m.shape}")
            print(f"  Sum of mask pixels: {np.sum(m)}")
        
        labeled_array, num_features = label(m)
        if debug:
            print(f"  Connected components found: {num_features}")
        
        if num_features == 0:
            if debug:
                print(f"  No pixels found for label '{label_name}'.")
            boxes[label_name] = None
        else:
            objects = find_objects(labeled_array)
            if debug:
                print(f"  Found objects (slices): {objects}")
            
            volumes = []
            for idx, obj in enumerate(objects):
                if obj is None:
                    volumes.append(0)
                    if debug:
                        print(f"  Object {idx} is None; volume set to 0.")
                else:
                    vol = (obj[0].stop - obj[0].start) * (obj[1].stop - obj[1].start) * (obj[2].stop - obj[2].start)
                    volumes.append(vol)
                    if debug:
                        print(f"  Object {idx}: slice {obj}, volume: {vol}")
            
            if len(volumes) == 0:
                if debug:
                    print(f"  No valid objects for label '{label_name}'.")
                boxes[label_name] = None
            else:
                idx = np.argmax(volumes)
                if debug:
                    print(f"  Largest object index: {idx} with volume {volumes[idx]}")
                obj = objects[idx]
                d_min, d_max = obj[0].start, obj[0].stop - 1
                h_min, h_max = obj[1].start, obj[1].stop - 1
                w_min, w_max = obj[2].start, obj[2].stop - 1
                if debug:
                    print(f"  Raw bounding box for label '{label_name}':")
                    print(f"    d_min: {d_min}, d_max: {d_max}, h_min: {h_min}, h_max: {h_max}, w_min: {w_min}, w_max: {w_max}")
                
                d_min = max(d_min - padding, 0)
                h_min = max(h_min - padding, 0)
                w_min = max(w_min - padding, 0)
                d_max = min(d_max + padding, m.shape[0]-1)
                h_max = min(h_max + padding, m.shape[1]-1)
                w_max = min(w_max + padding, m.shape[2]-1)
                if debug:
                    print(f"  Final bounding box for label '{label_name}' with padding {padding}:")
                    print(f"    (d_min: {d_min}, d_max: {d_max}, h_min: {h_min}, h_max: {h_max}, w_min: {w_min}, w_max: {w_max})")
                boxes[label_name] = (d_min, d_max, h_min, h_max, w_min, w_max)
    return boxes





def generate_and_save_pseudo_labels(args, data_loader, model, save_file, device, save_true_labels=False):
    sw_bs = args.test.sw_batch
    infer_overlap = args.test.infer_overlap
    bbox_padding = args.test.bbox_padding

    POINTS_PER_SLICE    = 20
    BG_POINTS_PER_SLICE = 20

    dice_metric = DiceMetric(include_background=True)
    hd_metric   = HausdorffDistanceMetric(include_background=True, percentile=95)

    pseudo_label_dict = {}
    model.eval()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Generating pseudo-labels"):
            pid    = data['patient_id'][0]
            image  = data['image'].to(device)
            pad    = data['pad_list']
            crop   = data.get('box_slice', None)

            pred = torch.sigmoid(inference(model, image, batch_size=sw_bs, overlap=infer_overlap))
            pred_np = (pred > 0.5).squeeze().cpu().numpy() 


            bboxes = compute_bounding_box(pred_np, padding=bbox_padding)

            point_coords = {'ET': {}, 'TC': {}, 'WT': {}}
            point_labels = {'ET': {}, 'TC': {}, 'WT': {}}
            for i, lbl_name in enumerate(['ET', 'TC', 'WT']):
                mask3d = pred_np[i] 
                D, H, W = mask3d.shape
                for z in range(D):
                    fg = np.argwhere(mask3d[z] == 1)
                    bg = np.argwhere(mask3d[z] == 0)
                    pts, lbs = [], []
                    if fg.size:
                        choose = np.random.choice(len(fg), min(POINTS_PER_SLICE, len(fg)), replace=False)
                        sel = fg[choose]
                        pts.extend(sel.tolist()); lbs.extend([1]*len(sel))
                    if bg.size:
                        choose = np.random.choice(len(bg), min(BG_POINTS_PER_SLICE, len(bg)), replace=False)
                        sel = bg[choose]
                        pts.extend(sel.tolist()); lbs.extend([0]*len(sel))
                    if pts:
                        xy = np.stack([np.array(pts)[:,1], np.array(pts)[:,0]], axis=1)
                        point_coords[lbl_name][z] = xy
                        point_labels[lbl_name][z] = np.array(lbs, dtype=np.int8)


            if save_true_labels and 'label' in data:
                gt = data['label'].to(device)
                true_np = gt.squeeze(0).cpu().numpy()
            else:
                et = tc = wt = avg = None
                true_np = None

            pseudo_label_dict[pid] = {
                'pseudo_label':  pred_np,
                'image':         image.squeeze().cpu().numpy(),
                'domain_label':  1,
                'pad_list':      pad,
                'crop_list':     crop,
                'true_label':    true_np,
                'bounding_box':  bboxes,
                'point_coords':  point_coords,
                'point_labels':  point_labels,
            }

    with open(save_file, 'wb') as f:
        pickle.dump(pseudo_label_dict, f)
    print(f"Saved pseudo-labels (and points) to {save_file}")




def create_model(cfg, in_channels, device):
    num_classes = 3
    spatial_size = 3
    arch = cfg.model.architecture.lower()
    if arch == "segres_net":
        model = SegResNet(spatial_dims=spatial_size,
                          init_filters=32,
                          in_channels=in_channels,
                          out_channels=num_classes,
                          dropout_prob=0.2,
                          blocks_down=(1, 2, 2, 4),
                          blocks_up=(1, 1, 1)).to(device)
    elif arch == "grlsegres_net":
        model = GRLSegResNet(spatial_dims=3,
                             init_filters=32,
                             in_channels=in_channels,
                             out_channels=num_classes,
                             dropout_prob=0.2,
                             blocks_down=(1, 2, 2, 4),
                             blocks_up=(1, 1, 1),
                             num_domains=1,
                             alpha=1.0).to(device)
    elif arch == 'graph_grl_segresnet':
        model = GraphGRLSegResNet(
            spatial_dims=3,
            init_filters=32,
            in_channels=1,
            out_channels=3,
            dropout_prob=0.2,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            num_domains=2,
            alpha=1.0
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture}")
    return model

@hydra.main(config_path = 'conf', config_name = 'configs', version_base = None)
def main(cfg: DictConfig):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    dataset_folder = cfg.dataset.dataset_folder

    in_channels = 1
    model = create_model(cfg, in_channels, device)

    ckpt = torch.load(cfg.test.weights, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)

    model.eval()


    weights_dir = os.path.dirname(cfg.test.weights)
    parent_dir = os.path.dirname(weights_dir)
    pseudo_labels_dir = os.path.join(parent_dir, 'pseudo_labels')

    if os.path.exists(pseudo_labels_dir):
        shutil.rmtree(pseudo_labels_dir)

    os.makedirs(pseudo_labels_dir, exist_ok=True)
    save_file = os.path.join(pseudo_labels_dir, 'pseudo_labels.pkl')


    from brats import get_domain_adaptation_datasets_train_selective
    _, target_dataset = get_domain_adaptation_datasets_train_selective(
        dataset_folder=dataset_folder,
        mode="train",
        target_size=(128, 128, 128),
        version="brats2020",
        seedAh = 0,
        chosen_modality_for_target = 't1',
        selected = 't2'
    )


    save_true_labels = True

    test_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=cfg.test.batch,
        shuffle=True,
        num_workers=cfg.test.workers,
        pin_memory=True
    )

    generate_and_save_pseudo_labels(cfg, test_loader, model, save_file, device, save_true_labels)
    print("Pseudo label generation complete!")

if __name__ == '__main__':
    main()
