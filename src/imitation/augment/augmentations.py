"""Additional GPU-accelerated image augmentations. When working with
image-based environments, these can be helpful for BC, and essential for the
GAIL discriminator."""

import torch as th
from kornia.color.gray import rgb_to_grayscale
from kornia.filters.gaussian import GaussianBlur2d
from torch import nn

from imitation.augment.color import apply_lab_jitter


class GaussianNoise(nn.Module):
    """Apply zero-mean Gaussian noise with a given standard deviation to input
    tensor."""

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = x + self.std * th.randn_like(x)
        out.clamp_(0.0, 1.0)
        return out


class GaussianBlur(nn.Module):
    """Blurs images."""

    def __init__(self, kernel_hw=5, sigma=1):
        super().__init__()
        assert kernel_hw >= 1 and (kernel_hw % 2) == 1
        sigma = float(sigma)
        assert sigma > 0
        self.blur_op = GaussianBlur2d((kernel_hw, kernel_hw), (sigma, sigma))

    def forward(self, images):
        return self.blur_op(images)


class CIELabJitter(nn.Module):
    """Apply 'jitter' in CIELab color space."""

    def __init__(self, max_lum_scale, max_uv_rads):
        super().__init__()
        self.max_lum_scale = max_lum_scale
        self.max_uv_rads = max_uv_rads

    def forward(self, x):
        # we take in stacked [N,C,H,W] images, where C=3*T. We then reshape
        # into [N,T,C,H,W] like apply_lab_jitter expects.
        stack_depth = x.size(1) // 3
        assert x.size(1) == 3 * stack_depth, x.shape
        x_reshape = x.view(x.shape[:1] + (stack_depth, 3) + x.shape[2:])
        jittered_reshape = apply_lab_jitter(
            x_reshape, self.max_lum_scale, self.max_uv_rads
        )
        jittered = jittered_reshape.view(x.shape)
        return jittered


class Rot90(nn.Module):
    """Apply a 0/90/180/270 degree rotation to the image."""

    def forward(self, images):
        batch_size = images.size(0)
        generated_rots = th.randint(low=0, high=4, size=(batch_size,))
        rot_elems = []
        for batch_idx in range(batch_size):
            rot_opt = generated_rots[batch_idx]
            rotated_image = th.rot90(images[batch_idx], k=rot_opt, dims=(1, 2))
            rot_elems.append(rotated_image)
        rot_images = th.stack(rot_elems)
        return rot_images


class Grayscale(nn.Module):
    __constants__ = ["p"]

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x):
        # separate out channels, just like in CIELabJitter
        batch_size = x.size(0)
        stack_depth = x.size(1) // 3
        assert x.size(1) == 3 * stack_depth, x.shape
        x_reshape = x.view(x.shape[:1] + (stack_depth, 3) + x.shape[2:])

        recol_elems = []
        rand_nums = th.rand(batch_size)
        for batch_idx in range(batch_size):
            if rand_nums[batch_idx] < self.p:
                recol_reduced = rgb_to_grayscale(x_reshape[batch_idx])
                rep_spec = (1,) * (recol_reduced.ndim - 3) + (3, 1, 1)
                recol = recol_reduced.repeat(rep_spec)
                recol_elems.append(recol)
            else:
                recol_elems.append(x_reshape[batch_idx])

        unshaped_result = th.stack(recol_elems, dim=0)
        result = unshaped_result.view(x.shape)

        return result
