import io

import pytest
import pandas as pd

from src.applications.services.utils.convert_zipfile import ZipFile


class TestCase:

    def test_valid(self):
        zipfile = ZipFile()
        with io.BytesIO() as buffer:
            expected = buffer
            zipfile.add_data(
                filename='test.csv',
                data=pd.DataFrame({'a': [0, 1, 2]}).to_csv()
            )
            actual = zipfile.main(buffer=buffer)
            assert isinstance(expected, io.BytesIO)
            assert isinstance(actual, io.BytesIO)
            assert expected == actual

    def test_valid_2(self):
        zipfile = ZipFile()
        with io.BytesIO() as buffer:
            expected = buffer
            zipfile.add_data(
                filename='test.csv',
                data='abc'
            )
            actual = zipfile.main(buffer=buffer)
            assert isinstance(expected, io.BytesIO)
            assert isinstance(actual, io.BytesIO)
            assert expected == actual

    def test_invalid(self):
        zipfile = ZipFile()
        with io.BytesIO() as buffer:
            zipfile.add_data(
                filename='test.csv',
                data=None
            )
            with pytest.raises(TypeError):
                zipfile.main(buffer=buffer)
