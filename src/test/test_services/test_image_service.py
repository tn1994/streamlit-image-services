import pytest

from src.applications.services.image_service import ImageService


class TestCase:
    image_service = ImageService()

    def test_is_exists_in_url_valid(self):
        expected = True

        url_list: list = [
            'http://hoge/foo/bar.jpg',
        ]

        for url in url_list:
            actual = self.image_service.is_exists_in_url(url=url)
            assert expected == actual

    def test_is_exists_in_url_invalid(self):
        expected = True

        url_list: list = [
            'http://hoge/foo/bar.tiff',
            'http://hoge/foo/bar.jpg/0',
            'http://hoge/foo/bar.bmp',
        ]

        with pytest.raises(AssertionError):
            for url in url_list:
                actual = self.image_service.is_exists_in_url(url=url)
                assert expected == actual
