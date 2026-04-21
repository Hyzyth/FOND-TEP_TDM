import torch
import numpy as np


def extract_ampl_phase_3d(fft_im):
    amp = torch.sqrt(fft_im[..., 0]**2 + fft_im[..., 1]**2)
    phase = torch.atan2(fft_im[..., 1], fft_im[..., 0])
    return amp, phase

def low_freq_mutate_3d(amp_src, amp_trg, L = 0.1):
    _, _, D, H, W = amp_src.size()
    b_val = int(np.floor(min(D, H, W) * L))

    center_d, center_h, center_w = D // 2, H // 2, W // 2
    d0, d1 = center_d - b_val, center_d + b_val + 1
    h0, h1 = center_h - b_val, center_h + b_val + 1
    w0, w1 = center_w - b_val, center_w + b_val + 1

    amp_src[:, :, d0:d1, h0:h1, w0:w1] = amp_trg[:, :, d0:d1, h0:h1, w0:w1]
    return amp_src




def FDA_source_to_target_3d(src_img, trg_img, L=0.1):

    fft_src = torch.fft.fftn(src_img, dim=(-3, -2, -1))
    fft_trg = torch.fft.fftn(trg_img, dim=(-3, -2, -1))
    
    src_center = torch.fft.fftshift(fft_src, dim=(-3, -2, -1))
    trg_center = torch.fft.fftshift(fft_trg, dim=(-3, -2, -1))
    
    amp_src = torch.abs(src_center)
    phase  = torch.angle(src_center)
    amp_trg = torch.abs(trg_center)
    
    B, C, D, H, W = amp_src.shape
    b_val = int(np.floor(min(D, H, W) * L))
    cd, ch, cw = D//2, H//2, W//2
    d0, d1 = cd - b_val, cd + b_val + 1
    h0, h1 = ch - b_val, ch + b_val + 1
    w0, w1 = cw - b_val, cw + b_val + 1
    
    amp_src[..., d0:d1, h0:h1, w0:w1] = amp_trg[..., d0:d1, h0:h1, w0:w1]
    
    src_center_new = amp_src * torch.exp(1j * phase)
    
    fft_src_new = torch.fft.ifftshift(src_center_new, dim=(-3, -2, -1))
    
    out = torch.fft.ifftn(fft_src_new, dim=(-3, -2, -1))
    return out.real


