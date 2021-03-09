"""Tools for working with color in the context of image augmentations."""
import enum
from typing import Tuple

import torch as th


class ColorSpace(str, enum.Enum):
    """Color space specification for use with image augmentations."""

    RGB = "RGB"
    GRAY = "GRAY"


@th.jit.script
def _lab_f(t: th.Tensor) -> th.Tensor:
    """Intermediate function used to convert from RGB to Lab."""
    delta = 6 / 29
    return th.where(t > delta ** 3, t ** (1 / 3), t / (3 * delta ** 2) + 4 / 29)


@th.jit.script
def _lab_f_inv(t: th.Tensor) -> th.Tensor:
    """Inverse of _lab_f."""
    delta = 6 / 29
    return th.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))


@th.jit.script
def rgb_to_lab(
    r: th.Tensor,
    g: th.Tensor,
    b: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """JITted conversion from RGB to Lab.

    Conversion formulas have been adapted from kornia.color.luv.rgb_to_luv so that:

    1. The function can be easily JITted.
    2. It converts to Lab instead of LUV."""

    # # Convert from Linear RGB to sRGB
    rs = th.where(r > 0.04045, th.pow(((r + 0.055) / 1.055), 2.4), r / 12.92)
    gs = th.where(g > 0.04045, th.pow(((g + 0.055) / 1.055), 2.4), g / 12.92)
    bs = th.where(b > 0.04045, th.pow(((b + 0.055) / 1.055), 2.4), b / 12.92)

    # sRGB to XYZ
    x = 0.412453 * rs + 0.357580 * gs + 0.180423 * bs
    y = 0.212671 * rs + 0.715160 * gs + 0.072169 * bs
    z = 0.019334 * rs + 0.119193 * gs + 0.950227 * bs

    # XYZ to Lab
    X_n = 0.950489
    Y_n = 1.000
    Z_n = 1.088840
    x_frac = _lab_f(x / X_n)
    y_frac = _lab_f(y / Y_n)
    z_frac = _lab_f(z / Z_n)
    L = 116 * y_frac - 16
    a = 500 * (x_frac - y_frac)
    b = 200 * (y_frac - z_frac)

    return L, a, b


@th.jit.script
def lab_to_rgb(
    L: th.Tensor,
    a: th.Tensor,
    b: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    # convert from Lab to XYZ
    X_n = 0.950489
    Y_n = 1.000
    Z_n = 1.088840
    L_16_frac = (L + 16) / 116.0
    x = X_n * _lab_f_inv(L_16_frac + a / 500.0)
    y = Y_n * _lab_f_inv(L_16_frac)
    z = Z_n * _lab_f_inv(L_16_frac - b / 200)

    # inlined call to rgb_to_xyz(th.stack((x, y, z)))
    # (note that these coefficients are from the matrix inverse of the
    # coefficients in rgb_to_lab; that is why they appear to have higher
    # precision)
    rs = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    gs = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    bs = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z

    # Convert from sRGB to RGB Linear
    r = th.where(rs > 0.0031308, 1.055 * th.pow(rs, 1 / 2.4) - 0.055, 12.92 * rs)
    g = th.where(gs > 0.0031308, 1.055 * th.pow(gs, 1 / 2.4) - 0.055, 12.92 * gs)
    b = th.where(bs > 0.0031308, 1.055 * th.pow(bs, 1 / 2.4) - 0.055, 12.92 * bs)

    # return th.stack((r, g, b), dim=-3)

    return r, g, b


@th.jit.script
def _unif_rand_range(lo: float, hi: float, size: int, device: th.device) -> th.Tensor:
    return (hi - lo) * th.rand((size,), device=device) + lo


@th.jit.script
def apply_lab_jitter(
    images: th.Tensor, max_lum_scale: float, max_uv_rads: float
) -> th.Tensor:
    """Apply random L*a*b* jitter to each element of a batch of images. The
    `images` tensor should be of shape `[B,...,C,H,W]`, where the ellipsis
    denotes extraneous dimensions which will not be transformed separately
    (e.g. there might be a time axis present after the batch axis)."""

    assert len(images.shape) >= 4 and images.shape[-3] == 3
    assert 2.0 >= max_lum_scale >= 1.0
    assert max_uv_rads >= 0.0

    L, a, b = rgb_to_lab(
        images[..., 0, :, :], images[..., 1, :, :], images[..., 2, :, :]
    )

    # random transforms
    batch_size = images.size(0)
    ab_angles = _unif_rand_range(-max_uv_rads, max_uv_rads, batch_size, images.device)
    lum_scale_factors = _unif_rand_range(
        1.0 / max_lum_scale, max_lum_scale, batch_size, images.device
    )
    sines = th.sin(ab_angles)
    cosines = th.cos(ab_angles)

    # resize transformations to take advantage of broadcasting
    new_shape = (batch_size,) + (1,) * (a.ndim - 1)
    sines = sines.view(new_shape)
    cosines = cosines.view(new_shape)
    lum_scale_factors = lum_scale_factors.view(new_shape)

    # now apply the transformations
    # (this is way faster than stacking and using th.matmul())
    trans_L = th.clamp(L * lum_scale_factors, 0.0, 100.0)
    trans_a = cosines * a - sines * b
    trans_b = sines * a + cosines * b

    trans_r, trans_g, trans_b = lab_to_rgb(trans_L, trans_a, trans_b)

    rgb_trans = th.stack((trans_r, trans_g, trans_b), dim=-3)

    # throw out colors that can't be expressed as RGB (this is probably not a
    # good way of doing it, but whatever)
    rgb_trans.clamp_(0.0, 1.0)

    return rgb_trans
