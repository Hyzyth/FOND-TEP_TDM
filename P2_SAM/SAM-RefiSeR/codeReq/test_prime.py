
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
from brats import get_datasets_prime, get_domain_adaptation_datasets_alwayst2_singletarget
from utils.meter import AverageMeter

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
import glob


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs("logger", exist_ok= True)
file_handler = logging.FileHandler(filename= "logger/logger_test.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Stream and file logging
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_value(value):
    """proprecess value to scaler"""
    if torch.is_tensor(value):
        return value.item()
    return value
  
def reconstruct_label(image):
    """reconstruct image label"""
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False)*(c3 == True)] = 2
    image[(c1 == True)*(c3 == True)] = 4
    return image

def inference(model, input, batch_size, overlap):
    """inference on input with trained model"""
    def _compute(input):
        return sliding_window_inference(inputs=input, roi_size=(128, 128, 128), sw_batch_size=batch_size, predictor=model, overlap=overlap)
    return _compute(input)


def test(args, data_loader, model, modality, gpu_id = None):

    """test the model on the test dataset"""
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    sw_bs = args.test.sw_batch
    infer_overlap = args.test.infer_overlap
    for data in tqdm(data_loader, desc = f"Modality {modality}"):
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        targets = data["label"].cuda()
        pad_list = data["pad_list"]
        inputs = inputs.cuda()
        model.cuda()
        with torch.no_grad():  
            if args.test.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2,)).flip(dims=(2,)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3,)).flip(dims=(3,)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(4,)).flip(dims=(4,)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3)).flip(dims=(2, 3)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 4)).flip(dims=(2, 4)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3, 4)).flip(dims=(3, 4)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3, 4)).flip(dims=(2, 3, 4)), batch_size=sw_bs, overlap=infer_overlap))
                predict = predict / 8.0 
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=sw_bs, overlap=infer_overlap))
                
        targets = targets[:, :, pad_list[-4]:targets.shape[2]-pad_list[-3], pad_list[-6]:targets.shape[3]-pad_list[-5], pad_list[-8]:targets.shape[4]-pad_list[-7]]
        predict = predict[:, :, pad_list[-4]:predict.shape[2]-pad_list[-3], pad_list[-6]:predict.shape[3]-pad_list[-5], pad_list[-8]:predict.shape[4]-pad_list[-7]]
        predict = (predict>0.5).squeeze()
        targets = targets.squeeze()
        dice_metrics = cal_dice(predict, targets, haussdor, meandice)
        confuse_metric = cal_confuse(predict, targets, patient_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = get_value(confuse_metric[0][0]), get_value(confuse_metric[1][0]), get_value(confuse_metric[2][0])
        et_spec, tc_spec, wt_spec = get_value(confuse_metric[0][1]), get_value(confuse_metric[1][1]), get_value(confuse_metric[2][1])
        metrics_dict.append(dict(id=patient_id,
            et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice, 
            et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
            et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
            et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec))
    save_seg_csv(metrics_dict, args, gpu_id, modality, append_mean = (gpu_id is None),
                 temp = (gpu_id is not None))

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
    elif arch == "unet3d":
        model = UNet3D(in_channels=in_channels, num_classes=num_classes).to(device)
    elif arch == "v_net":
        model = VNet(spatial_dims=spatial_size,
                     in_channels=in_channels,
                     out_channels=num_classes,
                     dropout_dim=1,
                     bias=False).to(device)
    elif arch == "attention_unet":
        from monai.networks.nets import AttentionUnet
        model = AttentionUnet(spatial_dims=spatial_size,
                              in_channels=in_channels,
                              out_channels=num_classes,
                              channels=(8, 16, 32, 64, 128),
                              strides=(2, 2, 2, 2)).to(device)
    elif arch == "resunet_pp":
        model = ResUnetPlusPlus(in_channels=in_channels, out_channels=num_classes).to(device)
    elif arch == "unet_r":
        model = UNETR(in_channels=in_channels,
                      out_channels=num_classes,
                      img_size=(128, 128, 128),
                      proj_type='conv',
                      norm_name='instance').to(device)
    elif arch == "swinunet_r":
        model = SwinUNETR(img_size=128,
                          in_channels=in_channels,
                          out_channels=num_classes,
                          feature_size=48,
                          drop_rate=0.1,
                          attn_drop_rate=0.2,
                          dropout_path_rate=0.1,
                          spatial_dims=spatial_size,
                          use_checkpoint=False,
                          use_v2=False).to(device)
    elif arch == "ux_net":
        model = UXNET(in_chans=in_channels,
                      out_chans=num_classes,
                      depths=[2, 2, 2, 2],
                      feat_size=[48, 96, 192, 384],
                      drop_path_rate=0,
                      layer_scale_init_value=1e-6,
                      spatial_dims=spatial_size).to(device)
    elif arch == "nn_former":
        model = nnFormer(crop_size=np.array([128, 128, 128]),
                         embedding_dim=96,
                         input_channels=in_channels,
                         num_classes=num_classes,
                         depths=[2, 2, 2, 2],
                         num_heads=[3, 6, 12, 24],
                         deep_supervision=False,
                         conv_op=nn.Conv3d,
                         patch_size=[4, 4, 4],
                         window_size=[4, 4, 8, 4]).to(device)
    elif arch == "seg_scnet":
        from thesis.models.v2.model import SegSCNet
        model = SegSCNet(in_channels=in_channels,
                         out_channels=num_classes,
                         feature_size=48,
                         hidden_size=384,
                         num_heads=4,
                         dims=[48, 96, 192, 384],
                         depths=[3, 3, 3, 3],
                         do_ds=False).to(device)
    elif arch == "scfe_net":
        from thesis.models.v3.model import SCFENet
        model = SCFENet(spatial_dims=spatial_size,
                        init_filters=32,
                        in_channels=in_channels,
                        out_channels=num_classes,
                        blocks_down=(1, 2, 2, 4),
                        blocks_up=(1, 1, 1),
                        gradient_checkpointing=True,
                        num_heads=4,
                        dropout_prob=0.2,
                        attn_dropout_rate=0.1,
                        do_ds=False,
                        positional_embedding="perceptron",
                        drop_path=True,
                        qkv_bias=False).to(device)
    elif arch == "grlsegres_net":
        model = GRLSegResNet(spatial_dims=3,
                             init_filters=32,
                             in_channels=in_channels,
                             out_channels=num_classes,
                             dropout_prob=0.2,
                             blocks_down=(1, 2, 2, 4),
                             blocks_up=(1, 1, 1),
                             num_domains=2,
                             alpha=1.0).to(device)
    elif arch == 'graph_grl_segresnet':
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
        raise ValueError(f"Unknown architecture: {cfg.model.architecture}")
    return model


def worker(gpu, cfg, test_dataset, model_state_dict, num_gpus, mod):
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu)

    logger.info(f"GPU {gpu}: Evaluating modality {mod}")
    in_channels = 1
    model = create_model(cfg, in_channels, device)
    model.load_state_dict(model_state_dict)
    model.eval()

    total = len(test_dataset)
    per_gpu = total // num_gpus
    start = gpu * per_gpu
    end = total if gpu == num_gpus - 1 else (gpu + 1) * per_gpu
    subset_indices = list(range(start, end))
    subset = torch.utils.data.Subset(test_dataset, subset_indices)
    test_loader = torch.utils.data.DataLoader(
        subset, batch_size = cfg.test.batch, shuffle = False,
        num_workers = cfg.test.workers, pin_memory = True
    )

    logger.info(f"GPU {gpu}: Processing indices {start} to {end}")
    test(cfg, test_loader, model, mod, gpu_id = gpu)


@hydra.main(config_path="conf", config_name="configs", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    dataset_dir = "/home/monetai/Desktop/dillan/allBrats/dataBrats"
    dataset_version = cfg.dataset.version

    modalities = ['t2']
    num_gpus = 1

    for mod in modalities:
        logger.info(f"Evaluating modality: {mod}")
        in_channels = 1

        ckpt = torch.load(cfg.test.weights, map_location=device)

        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        def strip_module_prefix(sd: dict) -> dict:
            new_sd = {}
            for k, v in sd.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_sd[new_key] = v
            return new_sd
        
        state_dict = strip_module_prefix(state_dict)


        model = create_model(cfg, in_channels, device)
        model.load_state_dict(state_dict, strict = False)
        model.eval()

        test_dataset = get_domain_adaptation_datasets_alwayst2_singletarget(
            dataset_folder=dataset_dir,
            mode="train_val",
            target_size=(128, 128, 128),
            version="brats2020",
            seedAh = 42,
            chosen_modality = 't1'
        )


        save_path = os.path.join(cfg.training.exp_name, "csv")
        os.makedirs(save_path, exist_ok=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.test.batch, shuffle=False,
            num_workers=cfg.test.workers, pin_memory=True)
        test(cfg, test_loader, model, mod, gpu_id=None)
    print('done!!')


if __name__ == '__main__':
    main()
