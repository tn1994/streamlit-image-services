import pytest

from src.applications.views.base_view import BaseView


class TestCase:

    def test_base_view_valid(self):
        class TestValid(BaseView):
            title: str = 'Valid'

            def main(self):
                pass

        test_valid = TestValid()
        test_valid.main()

    def test_base_view_invalid(self):
        with pytest.raises(NotImplementedError):
            class TestInvalid(BaseView):
                pass

            test_invalid = TestInvalid()
            test_invalid.main()

    def test_base_view_invalid_2(self):
        with pytest.raises(NotImplementedError):
            class TestInvalid2(BaseView):
                title: str = 'Invalid'

            test_invalid_2 = TestInvalid2()
            test_invalid_2.main()
