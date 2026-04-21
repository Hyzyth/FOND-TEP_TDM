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

import torch.multiprocessing as mp
from torch.backends import cudnn

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

try:
    from thesis.models.SegUXNet.model import SegUXNet
    from thesis.models.v2.model import SegSCNet
    from thesis.models.v3.model import SCFENet
except ModuleNotFoundError:
    print('model not available, please train with other models')
    # sys.exit(1)

from functools import partial

import hydra
from omegaconf import OmegaConf, DictConfig
import logging
import os
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
import torch
import sys

import torch
import os
from torch.utils.data.dataset import Dataset, ConcatDataset
from utils.all_utils import pad_or_crop_image, minmax, load_nii, pad_image_and_label, listdir, get_brats_folder
from math import comb
from copy import deepcopy
import numpy as np
import random
import pickle
from scipy.ndimage import label, find_objects

os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


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


sys.path.append("/home/monetai/Desktop/dillan/allBrats/code/brrr/SAM")
from segment_anything import sam_model_registry, SamPredictor

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

def init_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def compute_bounding_box(mask, padding=2, debug=True):

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


def update_teacher_EMA(teacher, student, ema_decay):

    with torch.no_grad():
        student_params = dict(student.named_parameters())
        for name, teacher_param in teacher.named_parameters():
            student_param = student_params[name]
            teacher_param.data.mul_(ema_decay).add_(1 - ema_decay, student_param.data)
    return teacher


def stage1_generate_predictions(bbox_padding, data_loader, model, predictions_file, device, save_true_labels = False):
    sw_bs = 2
    infer_overlap = 0.6

    predictions_dict = {}
    model.eval()

    for data in tqdm(data_loader, desc = "Stage 1: Generating pseudo-labels"):
        patient_id = data['patient_id'][0]
        image = data['image'].to(device)
        pad_list = data['pad_list']
        crop_list = data.get('box_slice', None)

        pred = torch.sigmoid(inference(model, image, batch_size = sw_bs, overlap = infer_overlap))
        pseudo_label = (pred > 0.5).squeeze().cpu().numpy()

        bounding_boxes = compute_bounding_box(pseudo_label, padding = bbox_padding, debug = False)
        image_np = image.squeeze().cpu().numpy()

        predictions_dict[patient_id] = {
            'pseudo_label': pseudo_label,
            'bounding_boxes': bounding_boxes,
            'image': image_np,
            'pad_list': pad_list,
            'crop_list': crop_list,
        }

        if save_true_labels:
            true_label_tensor = data['label'].to(device)
            true_label = true_label_tensor.squeeze(0).cpu().numpy()
            predictions_dict[patient_id]['true_label'] = true_label

        del image, pred

    with open(predictions_file, 'wb') as f:
        pickle.dump(predictions_dict, f)
    
    print(f"Stage 1 complete: Predictions (and associated bounding boxes) saved to {predictions_file}")

def stage2_refine_with_sam(predictions_file, final_save_file, device):
    with open(predictions_file, 'rb') as f:
        data_dict = pickle.load(f)

    sam_checkpoint = "/home/monetai/Desktop/dillan/allBrats/code/brrr/SAM/sam_vit_h_4b8939.pth"
    model_type     = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    channel_names  = ['ET','TC','WT']
    CONF_THRESHOLD = 0.90

    for pid, info in tqdm(data_dict.items(), desc="Stage 2: SAM refine"):
        image_np        = info['image']         
        bounding_boxes  = info['bounding_boxes']
        orig_pred       = info['pseudo_label']    
        C, D, H, W      = orig_pred.shape

        out_vol = orig_pred.copy()

        stats = { ch: {'refined':0, 'rejected':0, 'skipped':0} for ch in channel_names }

        for ch_idx, ch in enumerate(channel_names):
            box = bounding_boxes.get(ch, None)
            if box is None:
                stats[ch]['skipped'] = D
                continue

            d_min, d_max, h_min, h_max, w_min, w_max = box
            d_min = max(0, d_min); d_max = min(D-1, d_max)

            for d in range(D):
                if d < d_min or d > d_max:
                    stats[ch]['skipped'] += 1
                    continue

                slice_img    = image_np[d]
                rgb          = np.stack([slice_img]*3, axis=-1)
                predictor.set_image(rgb)

                box_np = np.array([w_min, h_min, w_max, h_max])[None, :]
                with torch.no_grad():
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box_np,
                        multimask_output=False
                    )
                conf = float(scores[0])
                if conf >= CONF_THRESHOLD:
                    out_vol[ch_idx, d] = masks[0]
                    stats[ch]['refined'] += 1
                else:
                    stats[ch]['rejected'] += 1

        info['sam_pseudo_label'] = out_vol
        info.pop('bounding_boxes', None)

        summary = []
        for ch in channel_names:
            s = stats[ch]
            summary.append(
                f"{ch}: refined {s['refined']}, "
                f"rejected {s['rejected']}, skipped {s['skipped']}"
            )
        print(f"Patient {pid} → " + "; ".join(summary))

    with open(final_save_file, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"Stage 2 complete: saved {len(data_dict)} patients to {final_save_file}")


from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandGaussianNoised, RandSpatialCropd, EnsureTyped
)

class PseudoTargetDataset(Dataset):
    def __init__(self, pickle_file, filtered_ids_file=None, roi_size=(128,128,128), spatial_size=(128,128,128)):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        self.patient_ids = list(self.data.keys())



        all_ids = list(self.data.keys())
        if filtered_ids_file and os.path.exists(filtered_ids_file):
            with open(filtered_ids_file, 'r') as f:
                keep_ids = {line.strip() for line in f if line.strip()}
            self.patient_ids = [pid for pid in all_ids if pid in keep_ids]
        else:
            self.patient_ids = all_ids



        self.transforms = Compose([
            EnsureTyped(keys=["image","label"]),
            RandFlipd(keys=["image","label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image","label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image","label"], spatial_axis=2, prob=0.5),
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
        ])

    def __len__(self):
        return len(self.patient_ids)


    def __getitem__(self, idx):
        pid    = self.patient_ids[idx]
        sample = self.data[pid]

        pseudo = sample['pseudo_label'].astype(np.float32)
        sam    = sample.get('sam_pseudo_label', None)

        if sam is not None:
            combined = np.stack([
                pseudo[0],       
                pseudo[1],      
                pseudo[2],     
            ], axis=0).astype(np.float32)
        else:
            combined = pseudo

        d = {
            "image":        sample['image'][None].astype(np.float32),
            "label":        combined,
            "domain_label": 1,
            "patient_id":   pid,
        }

        return d

