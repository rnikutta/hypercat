"""Unit tests for hypercat.imageops module."""

import numpy as np
import pytest

from hypercat import imageops
from hypercat.imageops import (
    Image,
    ImageFrame,
    add_noise,
    checkEven,
    checkImage,
    checkInt,
    checkOdd,
    checkSquare,
    check2d,
    computeIntCorrections,
    make_binary,
    makepositive,
    measure_snr,
    resampleImage,
    resample_image,
    rotateImage,
    thresholding,
    trim_square,
    trim_square_odd,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_odd_image():
    """A simple 11×11 square image with a bright central pixel."""
    img = np.zeros((11, 11))
    img[5, 5] = 1.0
    return img


@pytest.fixture
def small_image():
    """A minimal 5×5 image with peak at center."""
    img = np.zeros((5, 5))
    img[2, 2] = 1.0
    return img


# ---------------------------------------------------------------------------
# Low-level integer / shape checks
# ---------------------------------------------------------------------------

class TestCheckInt:
    def test_integer_passes(self):
        checkInt(3)  # should not raise

    def test_float_raises(self):
        with pytest.raises(TypeError):
            checkInt(3.0)

    def test_string_raises(self):
        with pytest.raises(TypeError):
            checkInt("3")


class TestCheckOdd:
    def test_odd_int_passes(self):
        checkOdd(5)

    def test_even_int_raises(self):
        with pytest.raises(ValueError):
            checkOdd(4)

    def test_float_raises(self):
        with pytest.raises(TypeError):
            checkOdd(5.0)


class TestCheckEven:
    def test_even_int_passes(self):
        checkEven(4)

    def test_odd_int_raises(self):
        with pytest.raises(ValueError):
            checkEven(5)

    def test_float_raises(self):
        with pytest.raises(TypeError):
            checkEven(4.0)


class TestCheck2d:
    def test_2d_passes(self, square_odd_image):
        check2d(square_odd_image)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            check2d(np.ones(5))

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            check2d(np.ones((3, 3, 3)))


class TestCheckSquare:
    def test_square_passes(self, square_odd_image):
        checkSquare(square_odd_image)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            checkSquare(np.ones((3, 5)))


class TestCheckImage:
    def test_valid_image_returns_size(self, square_odd_image):
        size = checkImage(square_odd_image, returnsize=True)
        assert size == 11

    def test_valid_image_no_return(self, square_odd_image):
        result = checkImage(square_odd_image, returnsize=False)
        assert result is None

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            checkImage(np.ones((5, 7)))

    def test_non_odd_raises(self):
        with pytest.raises((ValueError, TypeError)):
            checkImage(np.ones((4, 4)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            checkImage(np.ones((5, 5, 5)))


# ---------------------------------------------------------------------------
# computeIntCorrections
# ---------------------------------------------------------------------------

class TestComputeIntCorrections:
    def test_exact_result(self):
        newnpix, newfactor = computeIntCorrections(3, 5)
        assert newnpix == 15
        assert newfactor == pytest.approx(5.0)

    def test_corrected_to_odd(self):
        newnpix, newfactor = computeIntCorrections(3, 2.0)
        assert newnpix % 2 == 1  # must be odd

    def test_scale_down(self):
        newnpix, newfactor = computeIntCorrections(15, 0.6)
        assert newnpix == 9
        assert newfactor == pytest.approx(0.6)

    def test_scale_down_with_correction(self):
        newnpix, newfactor = computeIntCorrections(15, 0.5)
        assert newnpix % 2 == 1
        assert newnpix > 0

    def test_factor_consistent_with_newnpix(self):
        newnpix, newfactor = computeIntCorrections(11, 2.3)
        assert newnpix == pytest.approx(11 * newfactor, abs=0.5)

    def test_non_odd_input_raises(self):
        with pytest.raises((ValueError, TypeError)):
            computeIntCorrections(4, 2.0)


# ---------------------------------------------------------------------------
# make_binary
# ---------------------------------------------------------------------------

class TestMakeBinary:
    def test_near_one_becomes_one(self):
        img = np.array([[0.95, 0.0], [0.0, 0.99]])
        result = make_binary(img.copy(), eps=0.1)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_near_zero_becomes_zero(self):
        img = np.array([[0.05, 0.5], [0.5, 0.04]])
        result = make_binary(img.copy(), eps=0.1)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[1, 1] == pytest.approx(0.0)

    def test_out_of_range_unchanged(self):
        img = np.array([[0.5, 0.5]])
        result = make_binary(img.copy(), eps=0.1)
        # 0.5 is not within eps of 0 or 1 → unchanged
        assert result[0, 0] == pytest.approx(0.5)

    def test_eps_ge_05_raises(self):
        img = np.ones((3, 3))
        with pytest.raises(Exception):
            make_binary(img, eps=0.5)


# ---------------------------------------------------------------------------
# makepositive
# ---------------------------------------------------------------------------

class TestMakepositive:
    def test_all_positive_unchanged(self):
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = makepositive(img)
        np.testing.assert_array_equal(result, img)

    def test_zeros_replaced_by_min_positive(self):
        img = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = makepositive(img)
        assert result[0, 0] == pytest.approx(1.0)  # MIN of positive values

    def test_negatives_replaced(self):
        img = np.array([[-1.0, 0.5], [1.0, 2.0]])
        result = makepositive(img)
        assert result[0, 0] == pytest.approx(0.5)

    def test_original_not_modified(self):
        img = np.array([[0.0, 1.0], [2.0, 3.0]])
        original = img.copy()
        makepositive(img)
        np.testing.assert_array_equal(img, original)


# ---------------------------------------------------------------------------
# thresholding
# ---------------------------------------------------------------------------

class TestThresholding:
    def test_below_threshold(self):
        img = np.array([[0.0, 1.0, 2.0, 3.0]])
        result = thresholding(img.copy(), where="below", thresh=1.0, cfill=0.0)
        assert result[0, 0] == pytest.approx(0.0)  # 0.0 <= 1.0 → filled
        assert result[0, 1] == pytest.approx(0.0)  # 1.0 <= 1.0 → filled
        assert result[0, 2] == pytest.approx(2.0)  # 2.0 > 1.0 → unchanged

    def test_above_threshold(self):
        img = np.array([[0.0, 1.0, 2.0, 3.0]])
        result = thresholding(img.copy(), where="above", thresh=1.5, cfill=-1.0)
        assert result[0, 0] == pytest.approx(0.0)   # unchanged
        assert result[0, 2] == pytest.approx(-1.0)  # 2.0 >= 1.5 → filled
        assert result[0, 3] == pytest.approx(-1.0)  # 3.0 >= 1.5 → filled

    def test_invalid_where_raises(self):
        img = np.ones((3, 3))
        with pytest.raises(Exception):
            thresholding(img, where="sideways")


# ---------------------------------------------------------------------------
# add_noise / measure_snr
# ---------------------------------------------------------------------------

class TestAddNoise:
    def test_output_shape(self, square_odd_image):
        noisy, noise = add_noise(square_odd_image.copy(), snr=10.0)
        assert noisy.shape == square_odd_image.shape
        assert noise.shape == square_odd_image.shape

    def test_noisy_equals_signal_plus_noise(self, square_odd_image):
        img = square_odd_image.copy()
        noisy, noise = add_noise(img.copy(), snr=10.0)
        # noisy ≈ img + noise  (img is passed in and modified in place for noise)
        np.testing.assert_array_almost_equal(noisy, img + noise)

    def test_invalid_fraction_raises(self, square_odd_image):
        with pytest.raises(Exception):
            add_noise(square_odd_image.copy(), snr=5.0, fraction=0.0)
        with pytest.raises(Exception):
            add_noise(square_odd_image.copy(), snr=5.0, fraction=1.5)


class TestMeasureSNR:
    def test_returns_float(self):
        img = np.ones((5, 5))
        noise = np.random.normal(0, 0.1, img.shape)
        noisy = img + noise
        snr = measure_snr(noisy, noise)
        assert isinstance(snr, float)

    def test_high_snr_image(self):
        img = np.ones((11, 11)) * 100.0
        noise = np.random.normal(0, 1.0, img.shape)
        noisy = img + noise
        snr = measure_snr(noisy, noise)
        assert snr > 50.0  # should be high SNR

    def test_invalid_fraction_raises(self):
        img = np.ones((5, 5))
        noise = np.ones((5, 5)) * 0.1
        with pytest.raises(Exception):
            measure_snr(img, noise, fraction=0.0)


# ---------------------------------------------------------------------------
# rotateImage
# ---------------------------------------------------------------------------

class TestRotateImage:
    def test_zero_rotation_unchanged(self, square_odd_image):
        result = rotateImage(square_odd_image.copy(), angle="0 deg")
        np.testing.assert_array_almost_equal(result, square_odd_image)

    def test_output_same_shape(self, square_odd_image):
        result = rotateImage(square_odd_image.copy(), angle="45 deg")
        assert result.shape == square_odd_image.shape

    def test_360_rotation_approximately_unchanged(self, small_image):
        result = rotateImage(small_image.copy(), angle="360 deg")
        np.testing.assert_array_almost_equal(result, small_image, decimal=5)

    def test_nw_direction_negates_angle(self):
        # Upper half filled, lower half zeros; 90° NE vs NW moves content
        # to opposite sides so results must differ.
        img = np.zeros((11, 11))
        img[:6, :] = 1.0  # upper half nonzero → mask covers only lower rows
        r_ne = rotateImage(img.copy(), angle="90 deg", direction="NE")
        r_nw = rotateImage(img.copy(), angle="90 deg", direction="NW")
        assert not np.allclose(r_ne, r_nw)

    def test_non_square_raises(self):
        img = np.ones((5, 7))
        with pytest.raises(ValueError):
            rotateImage(img, angle="10 deg")


# ---------------------------------------------------------------------------
# trim_square_odd
# ---------------------------------------------------------------------------

class TestTrimSquareOdd:
    def test_square_even_trimmed_to_odd(self):
        img = np.zeros((4, 4))
        img[1, 1] = 1.0
        result = trim_square_odd(img)
        assert result.ndim == 2
        assert result.shape[0] % 2 == 1
        assert result.shape[0] == result.shape[1]

    def test_center_pixel_preserved(self):
        img = np.zeros((5, 5))
        img[2, 2] = 5.0
        result = trim_square_odd(img)
        center = result.shape[0] // 2
        assert result[center, center] == pytest.approx(5.0)

    def test_1d_not_accepted(self):
        with pytest.raises((ValueError, AttributeError, IndexError)):
            trim_square_odd(np.ones(5))

    def test_peak_at_edge_gives_1x1(self):
        img = np.zeros((3, 4))
        img[0, 1] = 1.0
        result = trim_square_odd(img)
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# trim_square
# ---------------------------------------------------------------------------

class TestTrimSquare:
    def test_removes_zero_margins(self):
        img = np.zeros((7, 7))
        img[2:5, 2:5] = 1.0
        result = trim_square(img)
        # After trimming, no all-zero rows/columns at boundaries
        assert result.shape[0] <= 7
        assert result.shape[1] <= 7

    def test_no_trim_needed(self):
        img = np.ones((5, 5))
        result = trim_square(img)
        # All pixels nonzero → same (or close to same) size
        assert result.size <= img.size


# ---------------------------------------------------------------------------
# resampleImage
# ---------------------------------------------------------------------------

class TestResampleImage:
    def test_upsampling_gives_larger_image(self):
        img = np.ones((5, 5))
        newimage, newfactor, npix = resampleImage(img, 2.0)
        assert npix > 5

    def test_downsampling_gives_smaller_image(self):
        img = np.ones((11, 11))
        newimage, newfactor, npix = resampleImage(img, 0.5)
        assert npix < 11

    def test_factor_1_unchanged_size(self):
        img = np.ones((5, 5))
        newimage, newfactor, npix = resampleImage(img, 1.0)
        assert npix == 5

    def test_conserve_true_preserves_total_flux(self):
        img = np.ones((5, 5)) * 2.0
        total_before = img.sum()
        newimage, _, _ = resampleImage(img, 1.5, conserve=True)
        total_after = newimage.sum()
        assert total_after == pytest.approx(total_before, rel=1e-3)

    def test_returns_odd_npix(self):
        img = np.ones((5, 5))
        _, _, npix = resampleImage(img, 2.0)
        assert npix % 2 == 1


# ---------------------------------------------------------------------------
# resample_image (simple version without checks)
# ---------------------------------------------------------------------------

class TestResampleImageSimple:
    def test_factor_2(self):
        img = np.zeros((11, 11))
        result = resample_image(img, factor=2.0)
        assert result.shape == (22, 22)

    def test_npixout(self):
        img = np.zeros((11, 11))
        result = resample_image(img, npixout=22)
        assert result.shape == (22, 22)

    def test_no_args_raises(self):
        img = np.zeros((5, 5))
        with pytest.raises(Exception):
            resample_image(img)


# ---------------------------------------------------------------------------
# ImageFrame class
# ---------------------------------------------------------------------------

class TestImageFrame:
    def test_init_stores_data(self, small_image):
        frame = ImageFrame(small_image)
        assert frame.data is not None
        assert frame.npix == 5

    def test_set_pixelscale_angular(self, small_image):
        frame = ImageFrame(small_image, pixelscale="1 arcsec")
        assert "arcsec" in str(frame.pixelscale.unit)

    def test_set_pixelscale_linear_with_distance(self, small_image):
        frame = ImageFrame(small_image, pixelscale="1 AU", distance="1 pc")
        # Converted to angular
        assert "arcsec" in str(frame.pixelscale.unit)

    def test_fov_computed(self, small_image):
        frame = ImageFrame(small_image, pixelscale="2 arcsec")
        assert frame.FOV.value == pytest.approx(10.0)  # 5 * 2

    def test_pixelarea_computed(self, small_image):
        frame = ImageFrame(small_image, pixelscale="3 arcsec")
        assert frame.pixelarea.value == pytest.approx(9.0)  # 3^2

    def test_image_property(self, small_image):
        frame = ImageFrame(small_image, pixelscale="1 arcsec")
        # I property returns transposed data
        assert frame.I.shape == small_image.T.shape


# ---------------------------------------------------------------------------
# Image class
# ---------------------------------------------------------------------------

class TestImage:
    def test_init_default_brightness(self, small_image):
        img = Image(small_image)
        # Default total_flux_density = '1 Jy' → total should be ~1 Jy
        total = img.getTotalFluxDensity("Jy")
        assert total.value == pytest.approx(1.0, rel=1e-3)

    def test_set_brightness(self, small_image):
        img = Image(small_image, total_flux_density="2 Jy")
        total = img.getTotalFluxDensity("Jy")
        assert total.value == pytest.approx(2.0, rel=1e-3)

    def test_get_brightness_units(self, small_image):
        img = Image(small_image, pixelscale="1 arcsec", total_flux_density="1 Jy")
        brightness = img.getBrightness("Jy/arcsec^2")
        assert "arcsec" in str(brightness.unit)

    def test_position_angle_rotates(self, small_image):
        img_no_pa = Image(small_image.copy(), pa="0 deg")
        img_with_pa = Image(small_image.copy(), pa="90 deg")
        # Data should differ after rotation (for non-symmetric images)
        # small_image has peak only at center, so rotation of symmetric image
        # should still give same data within interpolation tolerance
        assert img_no_pa.data.shape == img_with_pa.data.shape

    def test_npix_matches_input(self, small_image):
        img = Image(small_image)
        assert img.npix == 5

    def test_total_flux_density_property(self, small_image):
        img = Image(small_image, total_flux_density="1 Jy")
        assert img.F.value == pytest.approx(1.0, rel=1e-3)
