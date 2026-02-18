"""Unit tests for hypercat.utils module."""

import numpy as np
import pytest

try:
    from hypercat.utils import arrayify, get_rootdir, mirror_axis, seq2str
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "hypercat.utils not importable (tkinter unavailable in this environment)",
        allow_module_level=True,
    )


class TestGetRootdir:
    def test_returns_string(self):
        result = get_rootdir()
        assert isinstance(result, str)

    def test_ends_with_hypercat(self):
        result = get_rootdir()
        assert result.rstrip("/").endswith("hypercat")

    def test_is_absolute_path(self):
        import os
        result = get_rootdir()
        assert os.path.isabs(result)


class TestSeq2str:
    def test_string_elements(self):
        assert seq2str(("a", "b", "c"), " - ") == "a - b - c"

    def test_integer_elements(self):
        assert seq2str([1, 2, 3], ":") == "1:2:3"

    def test_default_separator(self):
        assert seq2str(("x", "y"), ",") == "x,y"

    def test_single_element(self):
        assert seq2str(("a",), "-") == "a"

    def test_mixed_types(self):
        assert seq2str((1, "b", 3.0), ",") == "1,b,3.0"


class TestArrayify:
    def test_single_element_default_shape(self):
        arr = arrayify(5)
        assert arr.shape == (1, 1)
        assert arr[0, 0] == 5

    def test_list_to_default_shape(self):
        arr = arrayify([1, 2, 3])
        assert arr.shape == (3, 1)

    def test_tuple_to_shape(self):
        arr = arrayify((1, 2, 3, 4), shape=(2, 2))
        assert arr.shape == (2, 2)

    def test_fill_single_element(self):
        arr = arrayify(7, shape=(1, 3), fill=True)
        assert arr.shape == (3, 1)
        assert all(arr.flat[i] == 7 for i in range(arr.size))

    def test_truncation_if_seq_too_long(self):
        # More elements than shape allows → truncate
        arr = arrayify([1, 2, 3, 4, 5], shape=(1, 3))
        assert arr.size == 3

    def test_padding_with_none_if_seq_too_short(self):
        arr = arrayify([1, 2], shape=(1, 4))
        flat = list(arr.flat)
        assert flat[2] is None
        assert flat[3] is None

    def test_direction_x_transposes(self):
        # direction='x' (default) transposes, so (1, ne) becomes (ne, 1)
        arr = arrayify((10, 20, 30))
        assert arr.shape == (3, 1)

    def test_direction_y_no_transpose(self):
        arr = arrayify((10, 20, 30), direction="y")
        assert arr.shape == (1, 3)


class TestMirrorAxis:
    def test_axis0(self):
        c = np.arange(9).reshape((3, 3))
        result = mirror_axis(c, axis=0)
        # Shape along axis 0 should be 2*3-1 = 5
        assert result.shape == (5, 3)
        # First 3 rows should be the mirror of last 3 rows
        np.testing.assert_array_equal(result[0], c[2])
        np.testing.assert_array_equal(result[1], c[1])
        np.testing.assert_array_equal(result[2], c[0])
        np.testing.assert_array_equal(result[3], c[1])
        np.testing.assert_array_equal(result[4], c[2])

    def test_axis1(self):
        c = np.arange(9).reshape((3, 3))
        result = mirror_axis(c, axis=1)
        # Shape along axis 1 should be 2*3-1 = 5
        assert result.shape == (3, 5)

    def test_default_axis(self):
        c = np.ones((3, 5))
        result = mirror_axis(c)
        # Default axis=-2 means second-to-last, which for 2D is axis 0
        assert result.shape[0] == 2 * 3 - 1

    def test_1d_input(self):
        c = np.array([1, 2, 3])
        result = mirror_axis(c, axis=0)
        assert result.shape == (5,)
        np.testing.assert_array_equal(result, [3, 2, 1, 2, 3])

    def test_values_symmetric(self):
        c = np.array([[1, 2], [3, 4], [5, 6]])
        result = mirror_axis(c, axis=0)
        # result should be symmetric around central row
        assert result.shape[0] == 5
        np.testing.assert_array_equal(result[0], result[4])
        np.testing.assert_array_equal(result[1], result[3])
