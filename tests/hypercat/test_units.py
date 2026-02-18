"""Unit tests for hypercat.units module."""

import numpy as np
import pytest
from astropy import units as u

from hypercat.units import UNITS, getQuantity, getValueUnit


class TestUnitsDict:
    def test_units_has_expected_categories(self):
        expected = {"ANGULAR", "LINEAR", "TEMPERATURE", "WAVE", "LUMINOSITY",
                    "BRIGHTNESS", "FLUXDENSITY", "CUNITS", "INVERSELINEAR"}
        assert expected.issubset(set(UNITS.keys()))

    def test_cunits_is_union_of_angular_and_linear(self):
        assert set(UNITS["CUNITS"]) == set(UNITS["ANGULAR"]) | set(UNITS["LINEAR"])

    def test_angular_contains_arcsec(self):
        assert "arcsec" in UNITS["ANGULAR"]

    def test_angular_contains_deg(self):
        assert "deg" in UNITS["ANGULAR"]

    def test_linear_contains_pc(self):
        assert "pc" in UNITS["LINEAR"]

    def test_fluxdensity_contains_jy(self):
        assert "Jy" in UNITS["FLUXDENSITY"]

    def test_inverselinear_prefix(self):
        for u_str in UNITS["INVERSELINEAR"]:
            assert u_str.startswith("1/")


class TestGetValueUnit:
    def test_simple_string_jy(self):
        val, unit = getValueUnit("1 Jy", UNITS["FLUXDENSITY"])
        assert val == pytest.approx(1.0)
        assert str(unit) == "Jy"

    def test_string_with_float(self):
        val, unit = getValueUnit("5.2 mJy", UNITS["FLUXDENSITY"])
        assert val == pytest.approx(5.2)

    def test_negative_value(self):
        val, unit = getValueUnit("-3 Jy", UNITS["FLUXDENSITY"])
        assert val == pytest.approx(-3.0)

    def test_no_space_between_value_and_unit(self):
        val, unit = getValueUnit("10arcsec", UNITS["CUNITS"])
        assert val == pytest.approx(10.0)
        assert "arcsec" in str(unit)

    def test_unrecognized_unit_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            getValueUnit("1 parsec", UNITS["FLUXDENSITY"])

    def test_quantity_instance_input(self):
        q = 3.0 * u.Unit("Jy")
        val, unit = getValueUnit(q, UNITS["FLUXDENSITY"])
        assert val == pytest.approx(3.0)

    def test_non_string_non_quantity_raises(self):
        with pytest.raises(AttributeError):
            getValueUnit(42, UNITS["FLUXDENSITY"])

    def test_angular_arcsec(self):
        val, unit = getValueUnit("30 arcsec", UNITS["CUNITS"])
        assert val == pytest.approx(30.0)

    def test_angular_deg(self):
        val, unit = getValueUnit("45 deg", UNITS["ANGULAR"])
        assert val == pytest.approx(45.0)


class TestGetQuantity:
    def test_returns_astropy_quantity(self):
        q = getQuantity("1 Jy", UNITS["FLUXDENSITY"])
        assert isinstance(q, u.Quantity)

    def test_value_preserved(self):
        q = getQuantity("2.5 mJy", UNITS["FLUXDENSITY"])
        assert q.value == pytest.approx(2.5)

    def test_unit_preserved(self):
        q = getQuantity("1 arcsec", UNITS["CUNITS"])
        assert "arcsec" in str(q.unit)

    def test_linear_unit(self):
        q = getQuantity("1 pc", UNITS["LINEAR"])
        assert "pc" in str(q.unit)

    def test_temperature(self):
        q = getQuantity("1500 K", UNITS["TEMPERATURE"])
        assert q.value == pytest.approx(1500.0)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            getQuantity("1 lightyear", UNITS["ANGULAR"])
