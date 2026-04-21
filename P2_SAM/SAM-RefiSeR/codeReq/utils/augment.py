"""
Data augmentations
"""
import torch
from torch import nn
from random import random, uniform
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast
from monai.transforms import RandAffined, RandAxisFlipd

import numpy as np
import random
from scipy.special import comb

# credit CKD-TransBTS. The default augmenter.
class DataAugmenterNullified(nn.Module):
    def __init__(self):
        super(DataAugmenterNullified,self).__init__()
        self.flip_dim = []
        self.zoom_rate = uniform(0.7, 1.0)
        self.sigma_1 = uniform(0.5, 1.5)
        self.sigma_2 = uniform(0.5, 1.5)
        self.image_zoom = Zoom(zoom=self.zoom_rate, mode="trilinear", padding_mode="constant")
        self.label_zoom = Zoom(zoom=self.zoom_rate, mode="nearest", padding_mode="constant")
        self.noisy = RandGaussianNoise(prob=1, mean=0, std=uniform(0, 0.33))
        self.blur = GaussianSharpen(sigma1=self.sigma_1, sigma2=self.sigma_2)
        self.contrast = AdjustContrast(gamma=uniform(0.65, 1.5))
    def forward(self, images, labels):
        with torch.no_grad():
            for b in range(images.shape[0]):
                image = images[b].squeeze(0)
                label = labels[b].squeeze(0)
                if random() < 0.15:
                    image = self.image_zoom(image)
                    label = self.label_zoom(label)


                # if random() < 0.5:
                #     image = torch.flip(image, dims=(1,))
                #     lable = torch.flip(lable, dims=(1,))
                # if random() < 0.5:
                #     image = torch.flip(image, dims=(2,))
                #     lable = torch.flip(lable, dims=(2,))
                # if random() < 0.5:
                #     image = torch.flip(image, dims=(3,))
                #     lable = torch.flip(lable, dims=(3,))

                for dim in range(1, len(image.shape)):  # Only flip valid dimensions
                    if random() < 0.5:
                        image = torch.flip(image, dims=(dim,))
                        label = torch.flip(label, dims=(dim,))
                
                if random() < 0.15:
                    image = self.noisy(image)
                if random() < 0.15:
                    image = self.blur(image)
                if random() < 0.15:
                    image = self.contrast(image)
                images[b] = image.unsqueeze(0)
                labels[b] = label.unsqueeze(0)
            return images, labels

# A more conservative augmenter than the one above.
class DataAugmenterNullified2(nn.Module):
    def __init__(self):
        super(DataAugmenterNullified2, self).__init__()

        self.zoom_prob = 0.15  
        self.zoom_min = 0.9
        self.zoom_max = 1.0
        zoom_val = uniform(self.zoom_min, self.zoom_max)

        self.flip_prob = 0.2  

        self.noise_prob = 0.15
        noise_std = uniform(0, 0.2)

        self.sharpen_prob = 0.1
        sigma1 = uniform(0.5, 1.0)
        sigma2 = uniform(0.5, 1.0)

        self.contrast_prob = 0.1
        gamma_val = uniform(0.8, 1.2)

        self.image_zoom = Zoom(zoom=zoom_val, mode="trilinear", padding_mode="constant")
        self.label_zoom = Zoom(zoom=zoom_val, mode="nearest", padding_mode="constant")
        self.noisy = RandGaussianNoise(prob=1, mean=0, std=noise_std)
        self.blur = GaussianSharpen(sigma1=sigma1, sigma2=sigma2)
        self.contrast = AdjustContrast(gamma=gamma_val)

    def forward(self, images, labels):
        with torch.no_grad():
            batch_size = images.shape[0]

            for b in range(batch_size):
                image = images[b].squeeze(0)
                label = labels[b].squeeze(0)

                if random() < self.zoom_prob:
                    image = self.image_zoom(image)
                    label = self.label_zoom(label)

                for dim in range(1, len(image.shape)):
                    if random() < self.flip_prob:
                        image = torch.flip(image, dims=(dim,))
                        label = torch.flip(label, dims=(dim,))

                if random() < self.noise_prob:
                    image = self.noisy(image)

                if random() < self.sharpen_prob:
                    image = self.blur(image)

                if random() < self.contrast_prob:
                    image = self.contrast(image)

                images[b] = image.unsqueeze(0)
                labels[b] = label.unsqueeze(0)

        return images, labels

#################################################

# The Bezier Curve Intensity Augmenter
def bernstein_poly(i, n, t):
    return comb(n, i) * (t**(n - i)) * ((1 - t)**i)

def bezier_curve(points, nTimes = 1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals

class DataAugmenter(nn.Module):
    def __init__(self, prob = 0.5, nTimes = 10000):
        super(DataAugmenter, self).__init__()
        self.prob = prob
        self.nTimes = nTimes

    def apply_bezier_intensity_transform(self, x):
        if random.random() > self.prob:
            return x
        
        # pick random control points in [0, 1] for the curve
        points = [
            [0.0, 0.0],
            [random.random(), random.random()],
            [random.random(), random.random()],
            [1.0, 1.0]
        ]

        xvals, yvals = bezier_curve(points, nTimes = self.nTimes)

        if random.random() < 0.5:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)

        x_min, x_max = x.min(), x.max()

        if x_min == x_max:
            return x
        
        x_norm = (x - x_min) / (x_max - x_min)

        x_norm_aug = np.interp(x_norm, xvals, yvals)

        x_aug = x_norm_aug * (x_max - x_min) + x_min

        return x_aug


    def forward(self, images, labels):
        with torch.no_grad():
            batch_size = images.shape[0]

            for b in range(batch_size):
                image = images[b]
                label = labels[b]

                new_channels = []
                for c in range(image.shape[0]):
                    image_c = image[c]

                    image_np = image_c.cpu().numpy()

                    image_np_aug = self.apply_bezier_intensity_transform(image_np)

                    new_channels.append(torch.from_numpy(image_np_aug).to(image.device))

                images[b] = torch.stack(new_channels, dim = 0)
                labels[b] = label

        return images, labels
    
###################################################

class AttnUnetAugmentation(nn.Module):
    def __init__(self):
      super(AttnUnetAugmentation, self).__init__()
      self.axial_prob = uniform(0.1, 0.6)
      self.affine_prob = uniform(0.1, 0.5)
      self.crop_prob = uniform(0.1, 0.5)
      self.axial_flips = RandAxisFlipd(keys=["image", "label"], prob=self.axial_prob)
      self.affine = RandAffined(
          keys=["image", "label"],
          mode=("bilinear", "nearest"),
          prob=self.affine_prob,
          shear_range=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1),
          padding_mode="border",
      )

    def forward(self, data):
      with torch.no_grad():
        data = self.affine(data)
        data = self.axial_flips(data)
        return data