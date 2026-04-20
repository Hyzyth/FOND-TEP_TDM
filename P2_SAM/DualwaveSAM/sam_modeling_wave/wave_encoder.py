# wave_encoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------  DWT（Haar，下采样x2） -----------------------
def dwt(x):
    # x: [B, C, H, W]，H/W 需为偶数
    x1 = x[:, :, 0::2, 0::2]  # (2i-1,2j-1)
    x2 = x[:, :, 1::2, 0::2]  # (2i,  2j-1)
    x3 = x[:, :, 0::2, 1::2]  # (2i-1,2j)
    x4 = x[:, :, 1::2, 1::2]  # (2i,  2j)

    LL = x1 + x2 + x3 + x4
    LH = -x1 - x3 + x2 + x4
    HL = -x1 + x3 - x2 + x4
    HH = x1 - x3 - x2 + x4
    return LL, LH, HL, HH

def wave_details(x):
    """返回当前层的细节系（LH,HL,HH）拼接：作为 Wave_qkv。尺寸为 (H/2, W/2)，通道=3C。"""
    LL, LH, HL, HH = dwt(x)
    return torch.cat([LH, HL, HH], dim=1)  # [B, 3C, H/2, W/2]



# ----------------------- 常用 Daubechies 分解低通系数（dec_lo） -----------------------
WAVELET_PRESETS = {
    "haar": [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],  # db1
    "db2":  [-0.12940952,  0.22414387,  0.83651630,  0.48296291],
    "db3":  [ 0.03522629, -0.08544127, -0.13501102, 0.45987750, 0.80689151, 0.33267055],
}


# ----------------------- 可切换小波基的 2D 单层分解（stride=2） -----------------------
# ---- 预置：仍内置 db1(haar)/db2/db3；其余从 PyWavelets 读取 ----
WAVELET_PRESETS = {
    "haar": [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],  # == db1
    "db2":  [-0.12940952,  0.22414387,  0.83651630,  0.48296291],
    "db3":  [ 0.03522629, -0.08544127, -0.13501102,  0.45987750, 0.80689151, 0.33267055],
}

class WaveletDecomp2D(nn.Module):
    """
    wavelet:
      - 内置: 'haar' | 'db2' | 'db3'
      - 直接用名: 'sym4' | 'bior4.4' | 'rbio3.5' | 'coif3' ...（自动从 PyWavelets 取 dec_lo/dec_hi）
      - 自定义: ('custom', dec_lo, dec_hi_or_None)
    normalize: 是否将 dec_lo/dec_hi 归一化到单位能量；对 biorth 建议 False 以保持原始幅度关系。
    """
    def __init__(self, channels: int, wavelet='haar', normalize=True, padding_mode='reflect'):
        super().__init__()
        self.channels = channels
        self.padding_mode = padding_mode

        dec_lo, dec_hi = self._resolve_filters(wavelet, normalize)

        # 构造 2D 核：LL=lo⊗lo, LH=lo⊗hi, HL=hi⊗lo, HH=hi⊗hi
        ll = torch.outer(dec_lo, dec_lo)
        lh = torch.outer(dec_lo, dec_hi)
        hl = torch.outer(dec_hi, dec_lo)
        hh = torch.outer(dec_hi, dec_hi)
        k = dec_lo.numel()

        # "same/2" 输出所需 padding：out = floor((H + 2p - k)/2 + 1) == H/2
        self.pad = (k - 1) // 2

        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,k,k]
        weight = weight.repeat(channels, 1, 1, 1)                   # [4C,1,k,k]
        self.register_buffer("weight", weight)

    @staticmethod
    def _qmf_from_lo(dec_lo: torch.Tensor) -> torch.Tensor:
        """正交小波用 QMF 从低通推高通：hi[n] = (-1)^n * lo[::-1][n]"""
        lo_rev = torch.flip(dec_lo, dims=[0])
        signs  = torch.tensor([(-1.0)**i for i in range(len(dec_lo))], dtype=torch.float32, device=dec_lo.device)
        return lo_rev * signs

    def _resolve_filters(self, wavelet, normalize):
        # 1) 预置/自定义
        if isinstance(wavelet, str) and wavelet.lower() in WAVELET_PRESETS:
            dec_lo = torch.tensor(WAVELET_PRESETS[wavelet.lower()], dtype=torch.float32)
            dec_hi = None  # 正交家族：可由 QMF 推导
        elif isinstance(wavelet, tuple) and wavelet[0] == 'custom':
            dec_lo = torch.as_tensor(wavelet[1], dtype=torch.float32)
            dec_hi = None if wavelet[2] is None else torch.as_tensor(wavelet[2], dtype=torch.float32)
        # 2) 其他命名（symN/coifN/biorX.Y/rbioX.Y/dmey...）-> PyWavelets
        elif isinstance(wavelet, str):
            try:
                import pywt  # pip install PyWavelets
            except Exception as e:
                raise ValueError(
                    f"wavelet='{wavelet}' 需要 PyWavelets 提供系数，请先安装: pip install PyWavelets；"
                    f"或改用 ('custom', dec_lo, dec_hi). 原始错误: {e}"
                )
            try:
                w = pywt.Wavelet(wavelet)
                # PyWavelets 提供分析滤波器 dec_lo/dec_hi（我们就是做分解）
                dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
                dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"无法加载小波 '{wavelet}'：{e}")
        else:
            raise ValueError("wavelet 必须是字符串或 ('custom', dec_lo, dec_hi_or_None)")

        # 归一化（可选）
        if normalize:
            dec_lo = dec_lo / torch.norm(dec_lo)
            if dec_hi is not None:
                dec_hi = dec_hi / torch.norm(dec_hi)

        # 若没给高通（正交小波场景），用 QMF 自动推导
        if dec_hi is None:
            dec_hi = self._qmf_from_lo(dec_lo)
            if normalize:
                dec_hi = dec_hi / torch.norm(dec_hi)

        return dec_lo, dec_hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=self.padding_mode)
        return F.conv2d(x, self.weight, bias=None, stride=2, padding=0, groups=self.channels)

    def details(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)               # [B,4C,H/2,W/2]
        C = self.channels
        # 通道分块：LL, LH, HL, HH
        return torch.cat([y[:, 1*C:2*C], y[:, 2*C:3*C], y[:, 3*C:4*C]], dim=1)  # [B,3C,H/2,W/2]

    


# ----------------------- 基础卷积块 -----------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=True, **kw)
        self.bn   = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ----------------------- 两段式注意力（单层用） -----------------------
# ----------------------- 两段式注意力（单层） -----------------------
class TwoStageWaveXAttn(nn.Module):
    """
    输入（同一尺度）: ct_feat, pet_feat  [B, C, H, W]
    使用 WaveletDecomp2D(details) 取 q/kv:
        Stage-1: q1=Wave(CT).details ->D,  k1/v1=Wave(PET).details ->D   => attn1
        Stage-2: q2=attn1,           k2/v2=Wave(CT).details  ->D         => attn2 -> 1x1 -> out_ch
    输出: [B, out_ch, H/2, W/2]
    """
    def __init__(self, in_ch, out_ch=None, heads=4, dim_head=32, wavelet='haar'):
        super().__init__()
        self.h = heads
        self.dh = dim_head
        self.D = heads * dim_head
        self.out_ch = out_ch or in_ch

        # 每层通道不同，需要对应的 Wavelet 分解器（可复用同一套系数）
        self.wave = WaveletDecomp2D(in_ch, wavelet=wavelet)

        self.q1 = nn.Conv2d(3*in_ch, self.D, 1)
        self.k1 = nn.Conv2d(3*in_ch, self.D, 1)
        self.v1 = nn.Conv2d(3*in_ch, self.D, 1)
        self.k2 = nn.Conv2d(3*in_ch, self.D, 1)
        self.v2 = nn.Conv2d(3*in_ch, self.D, 1)
        self.out = nn.Conv2d(self.D, self.out_ch, 1)

    @staticmethod
    def _attn(q, k, v, h, dh):
        B, D, H, W = q.shape
        N = H * W
        def reshape(x):
            return x.view(B, h, dh, N).transpose(2, 3).contiguous()  # [B,h,N,dh]
        q = reshape(q); k = reshape(k); v = reshape(v)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        a = a.softmax(dim=-1)
        out = a @ v
        out = out.transpose(2, 3).contiguous().view(B, h*dh, H, W)
        return out

    def forward(self, ct_feat, pet_feat):
        # Wave 细节（H/2, W/2）
        w_ct  = self.wave.details(ct_feat)   # [B,3C,H/2,W/2]
        w_pet = self.wave.details(pet_feat)  # [B,3C,H/2,W/2]

        # Stage-1
        attn1 = self._attn(self.q1(w_ct), self.k1(w_pet), self.v1(w_pet), self.h, self.dh)  # [B,D,H/2,W/2]
        # Stage-2
        attn2 = self._attn(attn1, self.k2(w_ct), self.v2(w_ct), self.h, self.dh)            # [B,D,H/2,W/2]

        return self.out(attn2)  # [B,out_ch,H/2,W/2]




class WaveInjectAdd(nn.Module):
    """
    轻量“细节注入”交互，不做注意力：
      Δ_ct2pet = Conv1x1( Wave(CT).details )   # [B,C,H/2,W/2]
      Δ_pet2ct = Conv1x1( Wave(PET).details )
      pet_p  <- pet_p + α_ct2pet * Δ_ct2pet
      ct_p   <-  ct_p + α_pet2ct * Δ_pet2ct
    """
    def __init__(self, in_ch, wavelet='haar', alpha=0.5, learnable_alpha=True, shared_alpha=False):
        super().__init__()
        self.wave = WaveletDecomp2D(in_ch, wavelet=wavelet)
        self.map  = nn.Conv2d(3*in_ch, in_ch, kernel_size=1, bias=False)

        if learnable_alpha:
            if shared_alpha:
                self.alpha_ct2pet = nn.Parameter(torch.tensor(float(alpha)))
                self.alpha_pet2ct = self.alpha_ct2pet  # 共享一个 α
            else:
                self.alpha_ct2pet = nn.Parameter(torch.tensor(float(alpha)))
                self.alpha_pet2ct = nn.Parameter(torch.tensor(float(alpha)))
        else:
            # 固定比例
            self.register_buffer('alpha_ct2pet', torch.tensor(float(alpha)))
            self.register_buffer('alpha_pet2ct', torch.tensor(float(alpha)))

        self.use_sigmoid = True  # 建议用 sigmoid 约束到 [0,1]

    def forward(self, ct_ds, pet_ds, ct_p, pet_p):
        # Wave 细节（与池化后的 p* 同分辨率）
        w_ct  = self.wave.details(ct_ds)    # [B,3C,H/2,W/2]
        w_pet = self.wave.details(pet_ds)   # [B,3C,H/2,W/2]
        d_ct  = self.map(w_ct)              # [B,C,H/2,W/2]  给 PET
        d_pet = self.map(w_pet)             # [B,C,H/2,W/2]  给 CT

        a12 = torch.sigmoid(self.alpha_ct2pet) if self.use_sigmoid else self.alpha_ct2pet
        a21 = torch.sigmoid(self.alpha_pet2ct) if self.use_sigmoid else self.alpha_pet2ct

        pet_p = pet_p + a12 * d_ct
        ct_p  = ct_p  + a21 * d_pet
        return ct_p, pet_p



# ----------------------- 主体：多层 DWT 交互的 WaveEncoder -----------------------
class WaveEncoder(nn.Module):
    """
    输入:  x[:,0:1]=CT, x[:,1:2]=PET  （[B, 2, H, W]）
    过程:  每一层 (stage0~stage3)：
            - 正常卷积得到 dsX，并 maxpool 得到 pX（尺寸/2）
            - 用 dsX 做 Wave 细节 (LH,HL,HH) 作为 qkv，按“两段式注意力”做 CT↔PET 交互
            - 交互结果 fusedX（尺寸/2，通道 = 当前层通道）分别回加到 pX_ct 与 pX_pet
            - 带着增强后的 pX_* 进入下一层卷积
          最深层 (stage4)：
            - 再做一次两段式交互得到 fused4（尺寸/2），上采样到 ds4 尺寸，作为最终输出
    输出:  fused_out （形状与 ds4 一致： [B, 16*nf, H/16, W/16]）
    """
    def __init__(self, in_channels=1, n_filters=16, heads=4, dim_head=32,
                 wavelet='haar', alpha=0.5, learnable_alpha=True, shared_alpha=False):
        super().__init__()
        nf = n_filters
        self.pool = nn.MaxPool2d(2, 2)
        self.img_size = 256

        # ---- CT 分支 ----
        self.ct0_1 = BasicConv2d(in_channels, nf,     kernel_size=5, padding=2)
        self.ct0_2 = BasicConv2d(nf,         nf,     kernel_size=3, padding=1)
        self.ct0_r = BasicConv2d(in_channels, nf,     kernel_size=1)
        self.ct1_1 = BasicConv2d(nf,         2*nf,   kernel_size=3, padding=1)
        self.ct1_2 = BasicConv2d(2*nf,       2*nf,   kernel_size=3, padding=1)
        self.ct1_r = BasicConv2d(nf,         2*nf,   kernel_size=1)
        self.ct2_1 = BasicConv2d(2*nf,       4*nf,   kernel_size=3, padding=1)
        self.ct2_2 = BasicConv2d(4*nf,       4*nf,   kernel_size=3, padding=1)
        self.ct2_r = BasicConv2d(2*nf,       4*nf,   kernel_size=1)
        self.ct3_1 = BasicConv2d(4*nf,       8*nf,   kernel_size=3, padding=1)
        self.ct3_2 = BasicConv2d(8*nf,       8*nf,   kernel_size=3, padding=1)
        self.ct3_r = BasicConv2d(4*nf,       8*nf,   kernel_size=1)
        self.ct4_1 = BasicConv2d(8*nf,       16*nf,  kernel_size=3, padding=1)
        self.ct4_2 = BasicConv2d(16*nf,      16*nf,  kernel_size=3, padding=1)
        self.ct4_r = BasicConv2d(8*nf,       16*nf,  kernel_size=1)

        # ---- PET 分支（对称） ----
        self.pt0_1 = BasicConv2d(in_channels, nf,     kernel_size=5, padding=2)
        self.pt0_2 = BasicConv2d(nf,         nf,     kernel_size=3, padding=1)
        self.pt0_r = BasicConv2d(in_channels, nf,     kernel_size=1)
        self.pt1_1 = BasicConv2d(nf,         2*nf,   kernel_size=3, padding=1)
        self.pt1_2 = BasicConv2d(2*nf,       2*nf,   kernel_size=3, padding=1)
        self.pt1_r = BasicConv2d(nf,         2*nf,   kernel_size=1)
        self.pt2_1 = BasicConv2d(2*nf,       4*nf,   kernel_size=3, padding=1)
        self.pt2_2 = BasicConv2d(4*nf,       4*nf,   kernel_size=3, padding=1)
        self.pt2_r = BasicConv2d(2*nf,       4*nf,   kernel_size=1)
        self.pt3_1 = BasicConv2d(4*nf,       8*nf,   kernel_size=3, padding=1)
        self.pt3_2 = BasicConv2d(8*nf,       8*nf,   kernel_size=3, padding=1)
        self.pt3_r = BasicConv2d(4*nf,       8*nf,   kernel_size=1)
        self.pt4_1 = BasicConv2d(8*nf,       16*nf,  kernel_size=3, padding=1)
        self.pt4_2 = BasicConv2d(16*nf,      16*nf,  kernel_size=3, padding=1)
        self.pt4_r = BasicConv2d(8*nf,       16*nf,  kernel_size=1)

        # —— 轻量交互：各层的 Wave 注入（不做注意力）——
        self.inj01 = WaveInjectAdd(nf,   wavelet, alpha, learnable_alpha, shared_alpha)
        self.inj12 = WaveInjectAdd(2*nf, wavelet, alpha, learnable_alpha, shared_alpha)
        self.inj23 = WaveInjectAdd(4*nf, wavelet, alpha, learnable_alpha, shared_alpha)
        self.inj34 = WaveInjectAdd(8*nf, wavelet, alpha, learnable_alpha, shared_alpha)
        # 最深层输出
        self.wave_out = TwoStageWaveXAttn(in_ch=16*nf, out_ch=16*nf, heads=heads, dim_head=dim_head, wavelet=wavelet)

    # ---- 编码器单支路 ----
    def _enc_stage0(self, x, conv1, conv2, res):
        ds0 = conv2(conv1(x)) + res(x)     # [B,nf,H,W]
        p0  = self.pool(ds0)               # [B,nf,H/2,W/2]
        return ds0, p0
    def _enc_block(self, px, c1, c2, rs):
        ds = c2(c1(px)) + rs(px)
        p  = self.pool(ds)
        return ds, p

    def forward(self, x):
            ct, pet = x[:,0:1], x[:,1:2]

            # Stage 0
            ct_ds0, ct_p0 = self._enc_stage0(ct,  self.ct0_1, self.ct0_2, self.ct0_r)
            pet_ds0, pet_p0 = self._enc_stage0(pet, self.pt0_1, self.pt0_2, self.pt0_r)
            ct_p0, pet_p0 = self.inj01(ct_ds0, pet_ds0, ct_p0, pet_p0)  # [1, 16, 128, 128]

            # Stage 1
            ct_ds1, ct_p1   = self._enc_block(ct_p0,  self.ct1_1, self.ct1_2, self.ct1_r)
            pet_ds1, pet_p1 = self._enc_block(pet_p0, self.pt1_1, self.pt1_2, self.pt1_r)
            ct_p1, pet_p1 = self.inj12(ct_ds1, pet_ds1, ct_p1, pet_p1)  # [1, 32, 64, 64]

            # Stage 2
            ct_ds2, ct_p2   = self._enc_block(ct_p1,  self.ct2_1, self.ct2_2, self.ct2_r)
            pet_ds2, pet_p2 = self._enc_block(pet_p1, self.pt2_1, self.pt2_2, self.pt2_r)
            ct_p2, pet_p2 = self.inj23(ct_ds2, pet_ds2, ct_p2, pet_p2)  # [1, 64, 32, 32]

            # Stage 3
            ct_ds3, ct_p3   = self._enc_block(ct_p2,  self.ct3_1, self.ct3_2, self.ct3_r)
            pet_ds3, pet_p3 = self._enc_block(pet_p2, self.pt3_1, self.pt3_2, self.pt3_r)
            ct_p3, pet_p3 = self.inj34(ct_ds3, pet_ds3, ct_p3, pet_p3)  # [1, 128, 16, 16]

            # Stage 4（最深）：卷积后做一次“两段式注意力”，作为输出
            ct_ds4, _  = self._enc_block(ct_p3,  self.ct4_1, self.ct4_2, self.ct4_r)
            pet_ds4, _ = self._enc_block(pet_p3, self.pt4_1, self.pt4_2, self.pt4_r)    # [1, 256, 16, 16]

            fused_deep_half = self.wave_out(ct_ds4, pet_ds4)  # [B,16nf,H/32,W/32]
            fused_out = F.interpolate(fused_deep_half, size=ct_ds4.shape[-2:],
                                    mode='bilinear', align_corners=False)  # [B,16nf,H/16,W/16]
            return pet_ds4+ct_ds4+fused_out


    # # 无 wave 注入
    # def forward(self, x):
    #     ct, pet = x[:,0:1], x[:,1:2]

    #     # Stage 0
    #     ct_ds0, ct_p0 = self._enc_stage0(ct,  self.ct0_1, self.ct0_2, self.ct0_r)
    #     pet_ds0, pet_p0 = self._enc_stage0(pet, self.pt0_1, self.pt0_2, self.pt0_r)

    #     # Stage 1
    #     ct_ds1, ct_p1   = self._enc_block(ct_p0,  self.ct1_1, self.ct1_2, self.ct1_r)
    #     pet_ds1, pet_p1 = self._enc_block(pet_p0, self.pt1_1, self.pt1_2, self.pt1_r)

    #     # Stage 2
    #     ct_ds2, ct_p2   = self._enc_block(ct_p1,  self.ct2_1, self.ct2_2, self.ct2_r)
    #     pet_ds2, pet_p2 = self._enc_block(pet_p1, self.pt2_1, self.pt2_2, self.pt2_r)

    #     # Stage 3
    #     ct_ds3, ct_p3   = self._enc_block(ct_p2,  self.ct3_1, self.ct3_2, self.ct3_r)
    #     pet_ds3, pet_p3 = self._enc_block(pet_p2, self.pt3_1, self.pt3_2, self.pt3_r)

    #     # Stage 4（最深）：卷积后做一次“两段式注意力”，作为输出
    #     ct_ds4, _  = self._enc_block(ct_p3,  self.ct4_1, self.ct4_2, self.ct4_r)
    #     pet_ds4, _ = self._enc_block(pet_p3, self.pt4_1, self.pt4_2, self.pt4_r)    # [1, 256, 16, 16]

    #     return pet_ds4+ct_ds4



if __name__ == "__main__":
    model = WaveEncoder(in_channels=1, n_filters=16, heads=4, dim_head=32)  
    x = torch.randn(1, 2, 256, 256)  # [B, (CT,PET), H, W]
    fused_feat = model(x)            # [1, 256, 16, 16]  (当 n_filters=16)
    print(fused_feat.shape)
