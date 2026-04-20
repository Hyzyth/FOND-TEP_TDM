from argparse import Namespace
import torch
import numpy as np
from build_sam_wave import sam_model_registry
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace as Namespace



class DualwaveSAM(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace()
        args.image_size = 256
        args.sam_checkpoint = None
        args.wavelet = 'haar'   # haar/db2/sym4/bior4.4

        # 1) 构建 SAM
        self.model = sam_model_registry["dual_wave"](args)

        # 是否需要梯度 True/False
        for p in self.model.prompt_encoder.parameters():
            p.requires_grad = False 
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        inputs = {'image': x, 'original_size': (x.shape[-2], x.shape[-1])}
        outputs = self.model(inputs, False)

        if "aux_pseudo_logits" in outputs:
            return outputs["masks"],outputs["aux_pseudo_logits"]
        else:
            return outputs["masks"],None







from torchinfo import summary

def format_bytes(n):
    return f"{n/1024/1024:.1f} MB ({n/1024/1024/1024:.3f} GB)"

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 生成输入数据
    x = torch.randn(1, 2, 256, 256, device=device)  # ct,pet

    # 初始化模型并放到对应device
    model = DualwaveSAM().to(device)
    model.eval()

    # 统计参数量 / FLOPs
    stats = summary(model, input_data=x, verbose=0)
    print(f"Total Params: {stats.total_params / 1e6:.2f}M")
    print(f"Total FLOPs: {stats.total_mult_adds / 1e9:.2f}G")

    # ===== 显存占用统计 =====
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        with torch.no_grad():   
            _ = model(x)
            torch.cuda.synchronize()

        peak_alloc = torch.cuda.max_memory_allocated(device)   # 峰值已分配
        peak_reserved = torch.cuda.max_memory_reserved(device) # 峰值已保留（缓存池）
        cur_alloc = torch.cuda.memory_allocated(device)
        cur_reserved = torch.cuda.memory_reserved(device)

        print("[GPU Mem] Peak allocated:", format_bytes(peak_alloc))
        print("[GPU Mem] Peak reserved :", format_bytes(peak_reserved))
        print("[GPU Mem] Current allocated:", format_bytes(cur_alloc))
        print("[GPU Mem] Current reserved :", format_bytes(cur_reserved))
    else:
        print("GPU 未启用，无法统计显存。")

    # 验证输出形状
    y,_ = model(x)
    print("Output shape:", y.shape)