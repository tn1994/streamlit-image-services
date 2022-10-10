import pytest

from src.applications.services.utils.re_util import get_only_number
from src.applications.services.utils.re_util import get_only_version


class TestCase:

    def test_get_only_number_valid(self):
        expected = '123456'
        actual = get_only_number(text='123456')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        assert expected == actual

    def test_get_only_number_valid_2(self):
        expected = '123456'
        actual = get_only_number(text='http://hoge/123456/foo')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        assert expected == actual

    def test_get_only_number_invalid(self):
        expected = 'hoge'
        actual = get_only_number(text='http://hoge/123456/foo')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        with pytest.raises(AssertionError):
            assert expected == actual

    def test_get_only_number_invalid_2(self):
        expected = '123456'
        actual = get_only_number(text='http://hoge/10/123456/foo')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        with pytest.raises(AssertionError):
            assert expected == actual

    def test_get_only_version_valid(self):
        expected = '1.0.0'
        actual = get_only_version(text='hoge==1.0.0')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        assert expected == actual

    def test_get_only_version_valid_2(self):
        expected = '0.1.0'
        actual = get_only_version(text='foo==0.1.0')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        assert expected == actual

    def test_get_only_version_valid_3(self):
        expected = '1.0.0rc9'
        actual = get_only_version(text='exceptiongroup==1.0.0rc9')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        assert expected == actual

    def test_get_only_version_valid_4(self):
        expected = '2022.4'
        actual = get_only_version(text='tzdata==2022.4')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        assert expected == actual

    def test_get_only_version_invalid(self):
        expected = '4.11.1'
        actual = get_only_version(text='beautifulsoup41==4.11.1')
        assert isinstance(expected, str)
        assert isinstance(actual, str)
        with pytest.raises(AssertionError):
            assert expected == actual
        assert '41' == actual
