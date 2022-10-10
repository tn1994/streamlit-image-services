import pytest


class TestCase:

    def test_valid(self):
        expected = True
        actual = True
        assert isinstance(expected, bool)
        assert isinstance(actual, bool)
        assert expected == actual

    def test_invalid(self):
        expected = True
        actual = False
        assert isinstance(expected, bool)
        assert isinstance(actual, bool)
        with pytest.raises(AssertionError):
            assert expected == actual
