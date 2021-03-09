"""Testing image augmentation routines."""
import kornia.augmentation as kornia_aug
import pytest
import torch as th

import imitation.augment.augmentations as im_aug
from imitation import augment
from imitation.augment import color


@pytest.fixture(params=list(augment.ColorSpace))
def color_space(request):
    return request.param


@pytest.fixture(params=[1, 7])
def fake_image(color_space, request):
    stack_depth = request.param
    if color_space == augment.ColorSpace.RGB:
        channels = stack_depth * 3
    else:
        channels = stack_depth
    images = th.empty((2, channels, 5, 5), dtype=th.float)
    images.uniform_()
    return images


def test_rgb_lab_inverse():
    with th.random.fork_rng():
        th.random.manual_seed(172)
        random_rgb_values = th.rand((3, 4, 5, 6, 7))
        random_r, random_g, random_b = random_rgb_values
        random_l, random_a, random_b = color.rgb_to_lab(random_r, random_g, random_b)
        return_rgb_values = th.stack(
            color.lab_to_rgb(random_l, random_a, random_b), dim=0
        )
        assert th.allclose(random_rgb_values, return_rgb_values, atol=1e-5)


def test_rgb_lab_reference():
    """Test that our colour conversion code matches colormath code.

    How these values were generated:

        from colormath import color_conversions, color_objects
        import numpy as np
        rng = np.random.RandomState(59674)
        rgb_vectors = rng.beta(0.5, 0.5, (5, 3))
        lab_vectors = []
        for rgb_vector in rgb_vectors:
            rgb_obj = color_objects.sRGBColor(*rgb_vector)
            lab = color_conversions.convert_color(rgb_obj, color_objects.LabColor)
            lab_vectors.append([lab.lab_l, lab.lab_a, lab.lab_b])
        print('RGB:', rgb_vectors.T.tolist())
        print('LAB:', np.asarray(lab_vectors).T.tolist())
    """
    rgb_ref_planes = th.tensor(
        [
            # r values
            [
                0.9418068084992004,
                0.4890588253044638,
                0.22335639525239961,
                0.483732376795571,
                0.2530335154840136,
            ],
            # g values
            [
                0.08528592856546649,
                0.0179925874670938,
                0.025094437338448315,
                0.9915773876894971,
                0.07782393242884299,
            ],
            # b values
            [
                0.9879167275310364,
                0.7983191795857149,
                0.0026318336475583235,
                0.12554925153656105,
                0.1746511369139697,
            ],
        ]
    )
    lab_ref_planes = th.tensor(
        [
            # l values
            [
                58.17368801388632,
                35.551626169126564,
                9.067557258309073,
                89.15618605609535,
                14.336397284145317,
            ],
            # a values
            [
                93.96218427416697,
                71.73489779522308,
                24.181605753266222,
                -67.77313140527936,
                24.771263734371608,
            ],
            # b values
            [
                -62.709696635427804,
                -72.80577956586154,
                13.895959354045745,
                81.69474250902672,
                -5.512797933382963,
            ],
        ]
    )
    converted_lab = th.stack(color.rgb_to_lab(*rgb_ref_planes), dim=0)
    # LAB values are on the order of -100 to 100, so rtol of 1e-4 and atol of
    # 1e-2 is not totally insane (but still not very good; there are probably
    # lots of problems that a numercist could pick out)
    assert th.allclose(converted_lab, lab_ref_planes, rtol=1e-4, atol=1e-2)


@pytest.mark.parametrize(
    "option_to_test", sorted(augment.StandardAugmentations.known_options())
)
def test_standard_augs_smoke(option_to_test, fake_image, color_space):
    if (
        "color" in option_to_test or "gray" in option_to_test
    ) and color_space != augment.ColorSpace.RGB:
        pytest.skip("color augmentations don't support grayscale")
    augmenter = augment.StandardAugmentations(
        **{option_to_test: True}, stack_color_space=color_space
    )
    assert len(augmenter.kornia_ops) == 1, augmenter.kornia_ops
    new_image = augmenter(fake_image)
    assert new_image.shape == fake_image.shape
    assert new_image.dtype == fake_image.dtype


def test_from_string_spec(fake_image, color_space):
    if color_space != augment.ColorSpace.RGB:
        pytest.skip("color augmentations don't support grayscale")
    augmenter = augment.StandardAugmentations.from_string_spec(
        "translate,rotate_ex,color_jitter", stack_color_space=color_space
    )
    augmenter_types = {type(mod) for mod in augmenter.kornia_ops}
    assert augmenter_types == {kornia_aug.RandomAffine, im_aug.CIELabJitter}
    new_image = augmenter(fake_image)
    assert new_image.shape == fake_image.shape
    assert new_image.dtype == fake_image.dtype
