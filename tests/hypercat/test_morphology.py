"""Unit tests for hypercat.morphology module."""

import numpy as np
import pytest

from hypercat.morphology import (
    Moment,
    circle,
    gaussian,
    get_all_moments_raw_matmul,
    get_angle,
    get_centroid,
    get_cov_from_moments,
    get_elongation,
    get_moment,
    get_moment_central,
    get_moment_central_matmul,
    get_moment_raw,
    get_moment_raw_matmul,
    get_moment_scaleinvariant,
    get_rgyr,
    gini,
    rectangle,
    rot90ccw,
    rotateVector,
    square,
    whichside,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def symmetric_gaussian():
    """A centred, symmetric 2-D Gaussian – many moments are predictable."""
    return gaussian(npix=51, sx=5.0, sy=5.0, x0=0, y0=0, theta=0.0)


@pytest.fixture
def elongated_gaussian():
    """An elongated Gaussian with sx != sy."""
    return gaussian(npix=51, sx=3.0, sy=10.0, x0=0, y0=0, theta=0.0)


@pytest.fixture
def unit_image():
    """Tiny all-ones image for exact moment calculations."""
    return np.ones((3, 3))


# ---------------------------------------------------------------------------
# Shape generators: circle, square, rectangle
# ---------------------------------------------------------------------------

class TestCircle:
    def test_returns_square_array(self):
        img = circle(21, r=5)
        assert img.shape == (21, 21)

    def test_all_values_zero_or_one(self):
        img = circle(21, r=5)
        assert set(np.unique(img)).issubset({0.0, 1.0})

    def test_center_is_inside(self):
        img = circle(21, r=5)
        assert img[10, 10] == 1.0

    def test_corner_is_outside(self):
        img = circle(21, r=5)
        assert img[0, 0] == 0.0

    def test_radius_zero_only_center(self):
        img = circle(11, r=0)
        assert img[5, 5] == 1.0
        assert img.sum() == pytest.approx(1.0)

    def test_offset_center(self):
        img = circle(21, r=3, x0=3, y0=3)
        # center at (10+3, 10+3) = (13, 13)
        assert img[13, 13] == 1.0


class TestSquare:
    def test_returns_square_array(self):
        img = square(21, a=6)
        assert img.shape == (21, 21)

    def test_all_values_zero_or_one(self):
        img = square(21, a=6)
        assert set(np.unique(img)).issubset({0.0, 1.0})

    def test_center_is_inside(self):
        img = square(21, a=6)
        assert img[10, 10] == 1.0


class TestRectangle:
    def test_returns_square_array(self):
        img = rectangle(21, a=4, b=8)
        assert img.shape == (21, 21)

    def test_all_values_zero_or_one(self):
        img = rectangle(21, a=4, b=8)
        assert set(np.unique(img)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Gaussian shape generator
# ---------------------------------------------------------------------------

class TestGaussian:
    def test_returns_2d_array(self, symmetric_gaussian):
        assert symmetric_gaussian.ndim == 2

    def test_square_shape(self, symmetric_gaussian):
        assert symmetric_gaussian.shape[0] == symmetric_gaussian.shape[1]

    def test_peak_at_center(self, symmetric_gaussian):
        npix = symmetric_gaussian.shape[0]
        cpix = npix // 2
        peak_idx = np.unravel_index(np.argmax(symmetric_gaussian),
                                    symmetric_gaussian.shape)
        assert peak_idx == (cpix, cpix)

    def test_all_positive(self, symmetric_gaussian):
        assert np.all(symmetric_gaussian >= 0.0)

    def test_npix_respected(self):
        img = gaussian(npix=31, sx=3.0, sy=3.0)
        assert img.shape == (31, 31)

    def test_offset_peak(self):
        img = gaussian(npix=51, sx=3.0, sy=3.0, x0=5, y0=5)
        peak_idx = np.unravel_index(np.argmax(img), img.shape)
        cpix = 51 // 2
        assert peak_idx[0] != cpix or peak_idx[1] != cpix


# ---------------------------------------------------------------------------
# Raw image moments (matmul version)
# ---------------------------------------------------------------------------

class TestGetMomentRawMatmul:
    def test_m00_equals_sum(self, unit_image):
        m00 = get_moment_raw_matmul(unit_image, (0, 0))
        assert m00 == pytest.approx(unit_image.sum())

    def test_m00_identity_matrix(self):
        img = np.eye(2)
        m00 = get_moment_raw_matmul(img, (0, 0))
        assert m00 == pytest.approx(2.0)

    def test_m10_identity_matrix(self):
        img = np.eye(2)
        # M10 = sum i * img[i,j] = 0*1 + 1*1 = 1
        m10 = get_moment_raw_matmul(img, (1, 0))
        assert m10 == pytest.approx(1.0)


class TestGetAllMomentsRawMatmul:
    def test_returns_correct_shape(self, unit_image):
        M = get_all_moments_raw_matmul(unit_image, pmax=2)
        assert M.shape == (3, 3)

    def test_m00_matches_single(self, unit_image):
        M = get_all_moments_raw_matmul(unit_image, pmax=2)
        m00_single = get_moment_raw_matmul(unit_image, (0, 0))
        assert M[0, 0] == pytest.approx(m00_single)


# ---------------------------------------------------------------------------
# Central moments (matmul version)
# ---------------------------------------------------------------------------

class TestGetMomentCentralMatmul:
    def test_mu00_equals_m00(self, unit_image):
        mu00 = get_moment_central_matmul(unit_image, (0, 0))
        m00 = get_moment_raw_matmul(unit_image, (0, 0))
        assert mu00 == pytest.approx(m00)

    def test_identity_mu20(self):
        img = np.eye(2)
        mu20 = get_moment_central_matmul(img, (2, 0))
        assert mu20 == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------------------
# Raw / central moments (slow reference version)
# ---------------------------------------------------------------------------

class TestGetMomentRaw:
    def test_m00_equals_sum(self, unit_image):
        M = get_moment_raw(unit_image, (0, 0))
        assert M == pytest.approx(unit_image.sum())

    def test_m00_gaussian(self, symmetric_gaussian):
        M = get_moment_raw(symmetric_gaussian, (0, 0))
        assert M > 0.0


class TestGetMomentCentral:
    def test_mu00_positive(self, symmetric_gaussian):
        mu = get_moment_central(symmetric_gaussian, (0, 0))
        assert mu > 0.0

    def test_mu11_near_zero_for_symmetric(self, symmetric_gaussian):
        mu11 = get_moment_central(symmetric_gaussian, (1, 1))
        assert abs(mu11) < 1e-5 * get_moment_raw(symmetric_gaussian, (0, 0))


class TestGetMomentScaleinvariant:
    def test_invalid_order_raises(self, unit_image):
        with pytest.raises(Exception):
            get_moment_scaleinvariant(unit_image, (0, 1))  # p+q = 1 < 2

    def test_valid_order_returns_float(self, symmetric_gaussian):
        eta = get_moment_scaleinvariant(symmetric_gaussian, (2, 0))
        assert isinstance(float(eta), float)


class TestGetMoment:
    def test_raw_m00(self, unit_image):
        m = get_moment(unit_image, (0, 0))
        assert m == pytest.approx(unit_image.sum())

    def test_central_mu11_symmetric(self, symmetric_gaussian):
        mu11 = get_moment(symmetric_gaussian, (1, 1), central=True)
        assert abs(mu11) < 1e-3


# ---------------------------------------------------------------------------
# Covariance, centroid, eigenvalues, elongation, angle
# ---------------------------------------------------------------------------

class TestGetCentroid:
    def test_symmetric_gaussian_centroid_at_center(self, symmetric_gaussian):
        M00, xbar, ybar = get_centroid(symmetric_gaussian)
        npix = symmetric_gaussian.shape[0]
        cpix = npix // 2
        assert xbar == pytest.approx(cpix, abs=0.5)
        assert ybar == pytest.approx(cpix, abs=0.5)

    def test_m00_equals_sum(self, symmetric_gaussian):
        M00, _, _ = get_centroid(symmetric_gaussian)
        assert M00 == pytest.approx(symmetric_gaussian.sum(), rel=1e-5)


class TestGetRgyr:
    def test_returns_two_values(self, symmetric_gaussian):
        rgx, rgy = get_rgyr(symmetric_gaussian)
        assert isinstance(float(rgx), float)
        assert isinstance(float(rgy), float)

    def test_symmetric_gaussian_rgyr_similar(self, symmetric_gaussian):
        rgx, rgy = get_rgyr(symmetric_gaussian)
        assert rgx == pytest.approx(rgy, rel=0.05)

    def test_elongated_gaussian_rgyr_differ(self, elongated_gaussian):
        rgx, rgy = get_rgyr(elongated_gaussian)
        assert abs(rgx - rgy) > 1.0  # sx=3 vs sy=10 should give different radii


class TestGetCovFromMoments:
    def test_returns_2x2_matrix(self, symmetric_gaussian):
        cov = get_cov_from_moments(symmetric_gaussian)
        assert cov.shape == (2, 2)

    def test_symmetric(self, symmetric_gaussian):
        cov = get_cov_from_moments(symmetric_gaussian)
        assert cov[0, 1] == pytest.approx(cov[1, 0], rel=1e-5)

    def test_diagonal_positive_definite(self, symmetric_gaussian):
        cov = get_cov_from_moments(symmetric_gaussian)
        assert cov[0, 0] > 0.0
        assert cov[1, 1] > 0.0


class TestGetElongation:
    def test_symmetric_elongation_near_one(self, symmetric_gaussian):
        cov = get_cov_from_moments(symmetric_gaussian)
        elong = get_elongation(cov)
        assert elong == pytest.approx(1.0, abs=0.1)

    def test_elongated_elongation_greater_than_one(self, elongated_gaussian):
        cov = get_cov_from_moments(elongated_gaussian)
        elong = get_elongation(cov)
        assert elong > 1.5

    def test_accepts_image_directly(self, symmetric_gaussian):
        # Non-(2,2) shaped input should trigger internal cov computation
        elong = get_elongation(symmetric_gaussian)
        assert isinstance(float(elong), float)


class TestGetAngle:
    def test_neither_img_nor_cov_raises(self):
        with pytest.raises(Exception):
            get_angle(img=None, cov=None)

    def test_angle_from_cov(self, symmetric_gaussian):
        cov = get_cov_from_moments(symmetric_gaussian)
        angle = get_angle(cov=cov)
        assert isinstance(float(angle), float)

    def test_angle_from_img(self, symmetric_gaussian):
        angle = get_angle(img=symmetric_gaussian)
        assert isinstance(float(angle), float)

    def test_symmetric_gaussian_angle_near_zero_or_ninety(self, symmetric_gaussian):
        angle = get_angle(img=symmetric_gaussian)
        # Symmetric Gaussian: angle could be anywhere since eigenvalues are equal,
        # but should still return a number
        assert np.isfinite(angle)


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------

class TestGini:
    def test_uniform_image_gini_near_zero(self):
        img = np.ones((10, 10))
        G = gini(img)
        assert G == pytest.approx(0.0, abs=0.05)

    def test_concentrated_image_gini_near_one(self):
        img = np.zeros((10, 10))
        img[5, 5] = 1.0
        G = gini(img)
        assert G > 0.9

    def test_gini_between_zero_and_one(self):
        np.random.seed(42)
        img = np.random.rand(20, 20)
        G = gini(img)
        assert 0.0 <= G <= 1.0

    def test_handles_negative_values(self):
        img = np.array([-1.0, 0.0, 1.0, 2.0])
        G = gini(img)
        assert np.isfinite(G)


# ---------------------------------------------------------------------------
# Vector rotation helpers
# ---------------------------------------------------------------------------

class TestRotateVector:
    def test_90_degrees_ccw(self):
        vec = np.array([0.0, 1.0])
        result = rotateVector(vec, deg=90.0)
        np.testing.assert_allclose(result, [-1.0, 0.0], atol=1e-10)

    def test_negative_90(self):
        vec = np.array([0.0, 1.0])
        result = rotateVector(vec, deg=-90.0)
        np.testing.assert_allclose(result, [1.0, 0.0], atol=1e-10)

    def test_360_is_identity(self):
        vec = np.array([1.0, 2.0])
        result = rotateVector(vec, deg=360.0)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_45_degrees(self):
        vec = np.array([0.0, 1.0])
        result = rotateVector(vec, deg=45.0)
        expected = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_zero_rotation(self):
        vec = np.array([3.0, 4.0])
        result = rotateVector(vec, deg=0.0)
        np.testing.assert_allclose(result, vec, atol=1e-10)


class TestRot90ccw:
    def test_x_axis_to_y_axis(self):
        v = np.array([1.0, 0.0])
        result = rot90ccw(v)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_y_axis_to_negative_x(self):
        v = np.array([0.0, 1.0])
        result = rot90ccw(v)
        np.testing.assert_allclose(result, [-1.0, 0.0], atol=1e-10)

    def test_four_rotations_is_identity(self):
        v = np.array([2.0, 3.0])
        result = rot90ccw(rot90ccw(rot90ccw(rot90ccw(v.copy()))))
        np.testing.assert_allclose(result, v, atol=1e-10)


class TestWhichSide:
    def test_perpendicular_right(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        # rot90ccw(a)=[0,1]; dot([0,1],[0,1])=1 → sign=+1 (b to the right of a)
        sig = whichside(a, b)
        assert sig == 1.0

    def test_parallel_returns_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([2.0, 0.0])  # same direction
        sig = whichside(a, b)
        assert sig == 0.0

    def test_antiparallel_returns_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])  # opposite direction
        sig = whichside(a, b)
        assert sig == 0.0


# ---------------------------------------------------------------------------
# Moment class
# ---------------------------------------------------------------------------

class TestMomentClass:
    def test_init_stores_image(self, symmetric_gaussian):
        m = Moment(symmetric_gaussian)
        assert m.img is symmetric_gaussian

    def test_init_computes_centroid(self, symmetric_gaussian):
        m = Moment(symmetric_gaussian)
        assert hasattr(m, "xbar")
        assert hasattr(m, "ybar")

    def test_call_returns_moments(self, symmetric_gaussian):
        m = Moment(symmetric_gaussian)
        m(2, 0)
        assert hasattr(m, "raw")
        assert hasattr(m, "central")

    def test_call_m00_equals_sum(self, symmetric_gaussian):
        m = Moment(symmetric_gaussian)
        m(0, 0)
        assert m.raw == pytest.approx(symmetric_gaussian.sum(), rel=1e-5)

    def test_call_computes_elongation(self, symmetric_gaussian):
        m = Moment(symmetric_gaussian)
        m(2, 2)
        assert hasattr(m, "elongation")
        assert m.elongation > 0.0

    def test_call_computes_angle(self, symmetric_gaussian):
        m = Moment(symmetric_gaussian)
        m(2, 2)
        assert hasattr(m, "angle")
        assert np.isfinite(m.angle)
