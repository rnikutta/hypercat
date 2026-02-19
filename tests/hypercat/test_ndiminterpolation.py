"""Unit tests for hypercat.ndiminterpolation module."""

import numpy as np
import pytest
from numpy import ma

from hypercat.ndiminterpolation import NdimInterpolation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_1d():
    """1-D linear data: y = x."""
    theta = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return theta, data


@pytest.fixture
def simple_2d():
    """2-D data: z = x + y on a regular grid."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    theta = [x, y]
    X, Y = np.meshgrid(x, y, indexing="ij")
    data = X + Y  # shape (3, 3)
    return theta, data


@pytest.fixture
def log_1d():
    """1-D data suitable for log-space interpolation (all positive)."""
    theta = [np.array([1.0, 2.0, 4.0, 8.0])]
    data = np.array([1.0, 2.0, 4.0, 8.0])  # y = x
    return theta, data


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestNdimInterpolationInit:
    def test_init_linear_mode(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        assert interp.order == 1
        assert interp.mode == "linear"

    def test_init_log_mode(self, log_1d):
        theta, data = log_1d
        interp = NdimInterpolation(data, theta, order=1, mode="log")
        assert interp.mode == "log"

    def test_init_stores_theta(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        assert len(interp.theta) == 1

    def test_init_invalid_order_raises(self, simple_1d):
        theta, data = simple_1d
        with pytest.raises(Exception, match="order"):
            NdimInterpolation(data, theta, order=2, mode="linear")

    def test_init_shape_mismatch_raises(self, simple_1d):
        theta, data = simple_1d
        bad_theta = [np.array([1.0, 2.0])]  # wrong size
        with pytest.raises(Exception):
            NdimInterpolation(data, bad_theta, order=1, mode="linear")

    def test_init_log_mode_masks_negatives(self):
        theta = [np.array([1.0, 2.0, 3.0])]
        data = np.array([1.0, -1.0, 3.0])  # has negative
        # Log mode masks non-positive values (masked_less_equal); should init ok
        interp = NdimInterpolation(data, theta, order=1, mode="log")
        assert interp.mode == "log"

    def test_init_2d(self, simple_2d):
        theta, data = simple_2d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        assert len(interp.theta) == 2


# ---------------------------------------------------------------------------
# serialize_vector
# ---------------------------------------------------------------------------

class TestSerializeVector:
    def test_scalars_remain_scalars(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        vec = interp.serialize_vector([2.0])
        assert isinstance(vec, tuple)
        assert vec[0] == 2.0

    def test_list_converted_to_tuple(self, simple_2d):
        theta, data = simple_2d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        vec = interp.serialize_vector([[1.5, 2.0], 15.0])
        assert isinstance(vec[0], tuple)

    def test_array_converted_to_tuple(self, simple_2d):
        theta, data = simple_2d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        vec = interp.serialize_vector([np.array([1.0, 2.0]), 10.0])
        assert isinstance(vec[0], tuple)

    def test_single_value_preserved(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        vec = interp.serialize_vector([3.5])
        assert vec[0] == 3.5


# ---------------------------------------------------------------------------
# Interpolation correctness (__call__)
# ---------------------------------------------------------------------------

class TestNdimInterpolationCall:
    def test_interpolate_at_exact_point_linear(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        result = interp([3.0])
        assert float(result) == pytest.approx(3.0, rel=1e-5)

    def test_interpolate_midpoint_linear(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        result = interp([1.5])
        assert float(result) == pytest.approx(1.5, rel=1e-3)

    def test_interpolate_2d_at_exact_point(self, simple_2d):
        theta, data = simple_2d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        result = interp([2.0, 20.0])
        # z = x + y = 2 + 20 = 22
        assert float(result) == pytest.approx(22.0, rel=1e-3)

    def test_interpolate_2d_midpoint(self, simple_2d):
        theta, data = simple_2d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        result = interp([1.5, 15.0])
        # Linear interp: 1.5 + 15 = 16.5
        assert float(result) == pytest.approx(16.5, rel=1e-2)

    def test_multiple_points_returns_array(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        result = interp([(1.0, 2.0, 3.0)])
        assert hasattr(result, "__len__")
        assert len(result) == 3

    def test_log_mode_exact_point(self, log_1d):
        theta, data = log_1d
        interp = NdimInterpolation(data, theta, order=1, mode="log")
        result = interp([2.0])
        assert float(result) == pytest.approx(2.0, rel=1e-3)

    def test_log_mode_midpoint(self, log_1d):
        theta, data = log_1d
        interp = NdimInterpolation(data, theta, order=1, mode="log")
        # Between x=2 (y=2) and x=4 (y=4): in log space midpoint
        result = interp([3.0])
        assert float(result) > 0  # must be positive

    def test_output_scalar_for_scalar_input(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        result = interp([2.5])
        # Scalar (0-d array or float)
        assert np.ndim(result) == 0 or result.size == 1


# ---------------------------------------------------------------------------
# get_coords
# ---------------------------------------------------------------------------

class TestGetCoords:
    def test_coords_shape_scalar_input(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        coords, shape_ = interp.get_coords((2.0,))
        assert coords.shape[0] == 1  # one parameter dimension
        assert coords.shape[1] == 1  # one coordinate set

    def test_coords_shape_multi_value(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        coords, shape_ = interp.get_coords(((1.0, 2.0, 3.0),))
        assert coords.shape[1] == 3  # three coordinate sets

    def test_pixel_coords_in_range(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        coords, _ = interp.get_coords((1.0,))  # first grid point
        assert coords[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_last_grid_point_pixel(self, simple_1d):
        theta, data = simple_1d
        interp = NdimInterpolation(data, theta, order=1, mode="linear")
        coords, _ = interp.get_coords((5.0,))  # last grid point
        assert coords[0, 0] == pytest.approx(4.0, abs=1e-10)
