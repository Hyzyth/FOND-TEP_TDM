# wave_encoder.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Haar DWT (2× downsampling wavelet decomposition)
# ============================================================
def dwt(x):
    """
    Perform one-level 2D Haar Discrete Wavelet Transform.

    Args:
        x: Tensor [B, C, H, W] (H and W must be even)

    Returns:
        LL: low-frequency component
        LH: horizontal detail
        HL: vertical detail
        HH: diagonal detail
    """

    # Subsampled spatial quadrants
    x1 = x[:, :, 0::2, 0::2]  # (2i-1, 2j-1)
    x2 = x[:, :, 1::2, 0::2]  # (2i,   2j-1)
    x3 = x[:, :, 0::2, 1::2]  # (2i-1, 2j)
    x4 = x[:, :, 1::2, 1::2]  # (2i,   2j)

    # Haar wavelet combinations
    LL = x1 + x2 + x3 + x4
    LH = -x1 - x3 + x2 + x4
    HL = -x1 + x3 - x2 + x4
    HH = x1 - x3 - x2 + x4

    return LL, LH, HL, HH


def wave_details(x):
    """
    Extract wavelet detail coefficients.

    Returns concatenated high-frequency components:
    LH + HL + HH → [B, 3C, H/2, W/2]
    Used as feature tokens for attention (Wave_qkv).
    """
    _, LH, HL, HH = dwt(x)
    return torch.cat([LH, HL, HH], dim=1)


# ============================================================
# Predefined wavelet filter banks (low-pass coefficients)
# ============================================================
WAVELET_PRESETS = {
    "haar": [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],  # db1 equivalent
    "db2":  [-0.12940952, 0.22414387, 0.83651630, 0.48296291],
    "db3":  [0.03522629, -0.08544127, -0.13501102,
             0.45987750, 0.80689151, 0.33267055],
}


# ============================================================
# Learnable wavelet decomposition layer (stride = 2)
# ============================================================
class WaveletDecomp2D(nn.Module):
    """
    General 2D wavelet decomposition module.

    Supports:
        - Built-in: haar / db2 / db3
        - PyWavelets names: symN, coifN, biorX.Y, rbioX.Y
        - Custom filters: ('custom', dec_lo, dec_hi)

    Output:
        [LL, LH, HL, HH] concatenated → [B, 4C, H/2, W/2]
    """

    def __init__(self, channels: int, wavelet='haar',
                 normalize=True, padding_mode='reflect'):
        super().__init__()
        self.channels = channels
        self.padding_mode = padding_mode

        dec_lo, dec_hi = self._resolve_filters(wavelet, normalize)

        # Build 2D separable filters
        ll = torch.outer(dec_lo, dec_lo)
        lh = torch.outer(dec_lo, dec_hi)
        hl = torch.outer(dec_hi, dec_lo)
        hh = torch.outer(dec_hi, dec_hi)

        k = dec_lo.numel()
        self.pad = (k - 1) // 2  # symmetric padding for stride-2 conv

        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        weight = weight.repeat(channels, 1, 1, 1)

        self.register_buffer("weight", weight)

    @staticmethod
    def _qmf_from_lo(dec_lo: torch.Tensor) -> torch.Tensor:
        """
        Quadrature Mirror Filter (QMF).

        Derives high-pass filter from low-pass:
        hi[n] = (-1)^n * reverse(lo[n])
        """
        lo_rev = torch.flip(dec_lo, dims=[0])
        signs = torch.tensor(
            [(-1.0) ** i for i in range(len(dec_lo))],
            dtype=torch.float32,
            device=dec_lo.device,
        )
        return lo_rev * signs

    def _resolve_filters(self, wavelet, normalize):
        """
        Resolve wavelet filters from:
            - preset dictionary
            - custom tuple
            - PyWavelets library
        """

        # -------------------------
        # 1) Built-in presets
        # -------------------------
        if isinstance(wavelet, str) and wavelet.lower() in WAVELET_PRESETS:
            dec_lo = torch.tensor(
                WAVELET_PRESETS[wavelet.lower()],
                dtype=torch.float32,
            )
            dec_hi = None

        # -------------------------
        # 2) Custom filters
        # -------------------------
        elif isinstance(wavelet, tuple) and wavelet[0] == "custom":
            dec_lo = torch.as_tensor(wavelet[1], dtype=torch.float32)
            dec_hi = None if wavelet[2] is None else torch.as_tensor(
                wavelet[2], dtype=torch.float32
            )

        # -------------------------
        # 3) PyWavelets-based filters
        # -------------------------
        elif isinstance(wavelet, str):
            try:
                import pywt
            except Exception as e:
                raise ValueError(
                    f"Wavelet '{wavelet}' requires PyWavelets. "
                    f"Install with: pip install PyWavelets. Error: {e}"
                )

            try:
                w = pywt.Wavelet(wavelet)
                dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
                dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Failed to load wavelet '{wavelet}': {e}")

        else:
            raise ValueError(
                "wavelet must be a string or ('custom', dec_lo, dec_hi)"
            )

        # Normalize filter energy
        if normalize:
            dec_lo = dec_lo / torch.norm(dec_lo)
            if dec_hi is not None:
                dec_hi = dec_hi / torch.norm(dec_hi)

        # If high-pass missing → derive via QMF
        if dec_hi is None:
            dec_hi = self._qmf_from_lo(dec_lo)
            if normalize:
                dec_hi = dec_hi / torch.norm(dec_hi)

        return dec_lo, dec_hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward wavelet decomposition.

        Output:
            [B, 4C, H/2, W/2]
        """
        if self.pad > 0:
            x = F.pad(
                x,
                (self.pad, self.pad, self.pad, self.pad),
                mode=self.padding_mode,
            )

        return F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=2,
            padding=0,
            groups=self.channels,
        )

    def details(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return only high-frequency components (LH, HL, HH).

        Output:
            [B, 3C, H/2, W/2]
        """
        y = self.forward(x)
        C = self.channels

        return torch.cat(
            [y[:, 1 * C:2 * C], y[:, 2 * C:3 * C], y[:, 3 * C:4 * C]],
            dim=1,
        )


# ============================================================
# Basic Conv-BN-ReLU block
# ============================================================
class BasicConv2d(nn.Module):
    """Conv2D → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=True, **kw)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ============================================================
# Two-stage cross-modal wavelet attention
# ============================================================
class TwoStageWaveXAttn(nn.Module):
    """
    Cross-modal wavelet attention block.

    Pipeline:
        Stage 1:
            Q = CT wave details
            K,V = PET wave details

        Stage 2:
            Q = Stage1 output
            K,V = CT wave details

    Output:
        [B, out_ch, H/2, W/2]
    """

    def __init__(self, in_ch, out_ch=None,
                 heads=4, dim_head=32, wavelet='haar'):
        super().__init__()

        self.h = heads
        self.dh = dim_head
        self.D = heads * dim_head
        self.out_ch = out_ch or in_ch

        self.wave = WaveletDecomp2D(in_ch, wavelet=wavelet)

        self.q1 = nn.Conv2d(3 * in_ch, self.D, 1)
        self.k1 = nn.Conv2d(3 * in_ch, self.D, 1)
        self.v1 = nn.Conv2d(3 * in_ch, self.D, 1)
        self.k2 = nn.Conv2d(3 * in_ch, self.D, 1)
        self.v2 = nn.Conv2d(3 * in_ch, self.D, 1)
        self.out = nn.Conv2d(self.D, self.out_ch, 1)

    @staticmethod
    def _attn(q, k, v, h, dh):
        """Scaled dot-product attention on spatial tokens."""
        B, D, H, W = q.shape
        N = H * W

        def reshape(x):
            return x.view(B, h, dh, N).transpose(2, 3).contiguous()

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        a = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        a = a.softmax(dim=-1)

        out = a @ v
        out = out.transpose(2, 3).contiguous().view(B, h * dh, H, W)
        return out

    def forward(self, ct_feat, pet_feat):
        """Apply two-stage cross-attention on wavelet features."""

        w_ct = self.wave.details(ct_feat)
        w_pet = self.wave.details(pet_feat)

        attn1 = self._attn(
            self.q1(w_ct),
            self.k1(w_pet),
            self.v1(w_pet),
            self.h,
            self.dh,
        )

        attn2 = self._attn(
            attn1,
            self.k2(w_ct),
            self.v2(w_ct),
            self.h,
            self.dh,
        )

        return self.out(attn2)


# ============================================================
# Lightweight additive wavelet fusion module
# ============================================================
class WaveInjectAdd(nn.Module):
    """
    Simple wavelet-based feature injection (no attention).

    CT → PET and PET → CT feature enhancement via residual addition.
    """

    def __init__(self, in_ch, wavelet='haar',
                 alpha=0.5, learnable_alpha=True,
                 shared_alpha=False):
        super().__init__()

        self.wave = WaveletDecomp2D(in_ch, wavelet=wavelet)
        self.map = nn.Conv2d(3 * in_ch, in_ch, kernel_size=1, bias=False)

        if learnable_alpha:
            if shared_alpha:
                self.alpha_ct2pet = nn.Parameter(torch.tensor(float(alpha)))
                self.alpha_pet2ct = self.alpha_ct2pet
            else:
                self.alpha_ct2pet = nn.Parameter(torch.tensor(float(alpha)))
                self.alpha_pet2ct = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha_ct2pet", torch.tensor(float(alpha)))
            self.register_buffer("alpha_pet2ct", torch.tensor(float(alpha)))

        self.use_sigmoid = True

    def forward(self, ct_ds, pet_ds, ct_p, pet_p):
        """Inject wavelet details between CT and PET streams."""

        w_ct = self.wave.details(ct_ds)
        w_pet = self.wave.details(pet_ds)

        d_ct = self.map(w_ct)
        d_pet = self.map(w_pet)

        a12 = torch.sigmoid(self.alpha_ct2pet) if self.use_sigmoid else self.alpha_ct2pet
        a21 = torch.sigmoid(self.alpha_pet2ct) if self.use_sigmoid else self.alpha_pet2ct

        pet_p = pet_p + a12 * d_ct
        ct_p = ct_p + a21 * d_pet

        return ct_p, pet_p


# ============================================================
# Full dual-stream WaveEncoder backbone
# ============================================================
class WaveEncoder(nn.Module):
    """
    Dual-stream encoder (CT + PET) with wavelet fusion.

    Structure:
        - Parallel CNN encoders (CT / PET)
        - Multi-stage downsampling
        - Wavelet feature injection at each scale
        - Final cross-modal attention fusion
    """

    def __init__(self, in_channels=1, n_filters=16,
                 heads=4, dim_head=32,
                 wavelet='haar',
                 alpha=0.5,
                 learnable_alpha=True,
                 shared_alpha=False):
        super().__init__()

        nf = n_filters
        self.pool = nn.MaxPool2d(2, 2)

        # ---------------- CT branch ----------------
        self.ct0_1 = BasicConv2d(in_channels, nf, kernel_size=5, padding=2)
        self.ct0_2 = BasicConv2d(nf, nf, kernel_size=3, padding=1)
        self.ct0_r = BasicConv2d(in_channels, nf, kernel_size=1)

        self.ct1_1 = BasicConv2d(nf, 2 * nf, kernel_size=3, padding=1)
        self.ct1_2 = BasicConv2d(2 * nf, 2 * nf, kernel_size=3, padding=1)
        self.ct1_r = BasicConv2d(nf, 2 * nf, kernel_size=1)

        self.ct2_1 = BasicConv2d(2 * nf, 4 * nf, kernel_size=3, padding=1)
        self.ct2_2 = BasicConv2d(4 * nf, 4 * nf, kernel_size=3, padding=1)
        self.ct2_r = BasicConv2d(2 * nf, 4 * nf, kernel_size=1)

        self.ct3_1 = BasicConv2d(4 * nf, 8 * nf, kernel_size=3, padding=1)
        self.ct3_2 = BasicConv2d(8 * nf, 8 * nf, kernel_size=3, padding=1)
        self.ct3_r = BasicConv2d(4 * nf, 8 * nf, kernel_size=1)

        self.ct4_1 = BasicConv2d(8 * nf, 16 * nf, kernel_size=3, padding=1)
        self.ct4_2 = BasicConv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)
        self.ct4_r = BasicConv2d(8 * nf, 16 * nf, kernel_size=1)

        # ---------------- PET branch ----------------
        self.pt0_1 = BasicConv2d(in_channels, nf, kernel_size=5, padding=2)
        self.pt0_2 = BasicConv2d(nf, nf, kernel_size=3, padding=1)
        self.pt0_r = BasicConv2d(in_channels, nf, kernel_size=1)

        self.pt1_1 = BasicConv2d(nf, 2 * nf, kernel_size=3, padding=1)
        self.pt1_2 = BasicConv2d(2 * nf, 2 * nf, kernel_size=3, padding=1)
        self.pt1_r = BasicConv2d(nf, 2 * nf, kernel_size=1)

        self.pt2_1 = BasicConv2d(2 * nf, 4 * nf, kernel_size=3, padding=1)
        self.pt2_2 = BasicConv2d(4 * nf, 4 * nf, kernel_size=3, padding=1)
        self.pt2_r = BasicConv2d(2 * nf, 4 * nf, kernel_size=1)

        self.pt3_1 = BasicConv2d(4 * nf, 8 * nf, kernel_size=3, padding=1)
        self.pt3_2 = BasicConv2d(8 * nf, 8 * nf, kernel_size=3, padding=1)
        self.pt3_r = BasicConv2d(4 * nf, 8 * nf, kernel_size=1)

        self.pt4_1 = BasicConv2d(8 * nf, 16 * nf, kernel_size=3, padding=1)
        self.pt4_2 = BasicConv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)
        self.pt4_r = BasicConv2d(8 * nf, 16 * nf, kernel_size=1)

        # Wavelet fusion modules
        self.inj01 = WaveInjectAdd(nf, wavelet, alpha, learnable_alpha, shared_alpha)
        self.inj12 = WaveInjectAdd(2 * nf, wavelet, alpha, learnable_alpha, shared_alpha)
        self.inj23 = WaveInjectAdd(4 * nf, wavelet, alpha, learnable_alpha, shared_alpha)
        self.inj34 = WaveInjectAdd(8 * nf, wavelet, alpha, learnable_alpha, shared_alpha)

        # Final cross-modal fusion
        self.wave_out = TwoStageWaveXAttn(
            in_ch=16 * nf,
            out_ch=16 * nf,
            heads=heads,
            dim_head=dim_head,
            wavelet=wavelet,
        )

    # ========================================================
    # Encoding helpers (CT/PET shared structure)
    # ========================================================
    def _enc_stage0(self, x, conv1, conv2, res):
        """First stage encoding block with residual connection."""
        ds0 = conv2(conv1(x)) + res(x)
        p0 = self.pool(ds0)
        return ds0, p0

    def _enc_block(self, px, c1, c2, rs):
        """Generic downsampling encoding block."""
        ds = c2(c1(px)) + rs(px)
        p = self.pool(ds)
        return ds, p

    # ========================================================
    # Forward pass
    # ========================================================
    def forward(self, x):
        """
        Args:
            x: [B, 2, H, W] (CT + PET channels)

        Returns:
            fused representation after cross-modal wavelet fusion
        """

        ct, pet = x[:, 0:1], x[:, 1:2]

        # Stage 0
        ct_ds0, ct_p0 = self._enc_stage0(ct, self.ct0_1, self.ct0_2, self.ct0_r)
        pet_ds0, pet_p0 = self._enc_stage0(pet, self.pt0_1, self.pt0_2, self.pt0_r)
        ct_p0, pet_p0 = self.inj01(ct_ds0, pet_ds0, ct_p0, pet_p0)

        # Stage 1
        ct_ds1, ct_p1 = self._enc_block(ct_p0, self.ct1_1, self.ct1_2, self.ct1_r)
        pet_ds1, pet_p1 = self._enc_block(pet_p0, self.pt1_1, self.pt1_2, self.pt1_r)
        ct_p1, pet_p1 = self.inj12(ct_ds1, pet_ds1, ct_p1, pet_p1)

        # Stage 2
        ct_ds2, ct_p2 = self._enc_block(ct_p1, self.ct2_1, self.ct2_2, self.ct2_r)
        pet_ds2, pet_p2 = self._enc_block(pet_p1, self.pt2_1, self.pt2_2, self.pt2_r)
        ct_p2, pet_p2 = self.inj23(ct_ds2, pet_ds2, ct_p2, pet_p2)

        # Stage 3
        ct_ds3, ct_p3 = self._enc_block(ct_p2, self.ct3_1, self.ct3_2, self.ct3_r)
        pet_ds3, pet_p3 = self._enc_block(pet_p2, self.pt3_1, self.pt3_2, self.pt3_r)
        ct_p3, pet_p3 = self.inj34(ct_ds3, pet_ds3, ct_p3, pet_p3)

        # Stage 4 (deep fusion)
        ct_ds4, _ = self._enc_block(ct_p3, self.ct4_1, self.ct4_2, self.ct4_r)
        pet_ds4, _ = self._enc_block(pet_p3, self.pt4_1, self.pt4_2, self.pt4_r)

        fused_half = self.wave_out(ct_ds4, pet_ds4)

        fused_out = F.interpolate(
            fused_half,
            size=ct_ds4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return pet_ds4 + ct_ds4 + fused_out


# ============================================================
# Optional ablation: no wavelet injection version
# ============================================================
"""
Disabled baseline version:
- Same CNN backbone
- Wavelet fusion removed
- Used for ablation comparison
"""


# ============================================================
# Sanity check
# ============================================================
if __name__ == "__main__":
    model = WaveEncoder(in_channels=1, n_filters=16, heads=4, dim_head=32)
    x = torch.randn(1, 2, 256, 256)
    fused_feat = model(x)
    print(fused_feat.shape)
