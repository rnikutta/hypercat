"""Unit tests for hypercat.loggers module."""

import logging

import pytest

from hypercat.loggers import LogFormatter


class TestLogFormatter:
    def test_is_logging_formatter(self):
        fmt = LogFormatter()
        assert isinstance(fmt, logging.Formatter)

    def test_format_debug_record(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="debug message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "debug message" in result

    def test_format_info_record(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="info message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "info message" in result

    def test_format_warning_record(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warning message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "warning message" in result

    def test_format_error_record(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error message",
            args=(),
            exc_info=None,
        )
        result = fmt.format(record)
        assert "error message" in result

    def test_format_returns_string(self):
        fmt = LogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        assert isinstance(fmt.format(record), str)
