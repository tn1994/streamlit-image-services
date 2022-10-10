import pytest

from src.applications.services.torch_services.torch_sevice import seed


class TestCase:
    def test_seed_valid(self):
        expected = 42
        actual = seed
        assert isinstance(expected, int)
        assert isinstance(actual, int)
        assert expected == actual
