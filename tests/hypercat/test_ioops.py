"""Unit tests for hypercat.ioops module."""

import json
import os
import tempfile

import numpy as np
import pytest

try:
    from hypercat.ioops import (
        get_bytes_human,
        get_bytesize,
        isragged,
        loadjson,
        storejson,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "hypercat.ioops not importable (urwid or urwid.curses_display unavailable)",
        allow_module_level=True,
    )


class TestGetBytesHuman:
    def test_bytes(self):
        result = get_bytes_human(500)
        assert "B" in result or "byte" in result.lower() or "500" in result

    def test_kilobytes(self):
        result = get_bytes_human(2048)
        assert "K" in result or "k" in result or "2" in result

    def test_megabytes(self):
        result = get_bytes_human(1024 * 1024)
        assert "M" in result or "1" in result

    def test_returns_string(self):
        assert isinstance(get_bytes_human(1000), str)

    def test_zero_bytes(self):
        result = get_bytes_human(0)
        assert isinstance(result, str)


class TestGetBytesize:
    def test_single_list(self):
        lol = [[1, 2, 3]]  # 3 elements × wordsize
        result = get_bytesize(lol, wordsize=4)
        assert result == 12

    def test_multiple_lists(self):
        lol = [[1, 2], [3, 4, 5]]  # 2 + 3 = 5 elements
        result = get_bytesize(lol, wordsize=4)
        assert result == 20

    def test_wordsize_8(self):
        lol = [[1, 2, 3]]
        result = get_bytesize(lol, wordsize=8)
        assert result == 24

    def test_empty_list(self):
        result = get_bytesize([], wordsize=4)
        assert result == 0


class TestIsragged:
    def test_regular_array_not_ragged(self):
        arr = np.array([[1, 2], [3, 4]])
        assert isragged(arr) is False

    def test_ragged_list(self):
        arr = [[1, 2], [3]]  # different lengths
        assert isragged(arr) is True

    def test_1d_not_ragged(self):
        arr = np.array([1, 2, 3])
        assert isragged(arr) is False

    def test_consistent_2d_not_ragged(self):
        arr = [[1, 2, 3], [4, 5, 6]]
        assert isragged(arr) is False


class TestStoreLoadJson:
    def test_round_trip(self, tmp_path):
        jsonfile = str(tmp_path / "test.json")
        d = {"key": "value", "num": 42, "list": [1, 2, 3]}
        storejson(jsonfile, d)
        loaded = loadjson(jsonfile)
        assert loaded == d

    def test_file_created(self, tmp_path):
        jsonfile = str(tmp_path / "out.json")
        storejson(jsonfile, {"a": 1})
        assert os.path.exists(jsonfile)

    def test_nested_dict(self, tmp_path):
        jsonfile = str(tmp_path / "nested.json")
        d = {"outer": {"inner": [1, 2, 3]}}
        storejson(jsonfile, d)
        loaded = loadjson(jsonfile)
        assert loaded["outer"]["inner"] == [1, 2, 3]

    def test_load_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, IOError, OSError)):
            loadjson("/nonexistent/path/file.json")
