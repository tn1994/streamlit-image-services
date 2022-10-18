import logging
from functools import cache
from typing import Final
from contextlib import suppress

from py3pin.Pinterest import Pinterest

from ..utils.re_util import get_only_number
from ...models.pinterest_query_model import QueryModel

logger = logging.getLogger(__name__)


class PinterestService:
    """
    ref: https://github.com/bstoilov/py3-pinterest
    """

    pin_id: str = None
    board_id: str = None

    image_info_list: list = []
    pin_id_list: list = []
    board_id_list: list = []
    board_pin_count_list: list = []

    query_category_dict: Final[dict] = QueryModel(
    ).get_query()  # todo: change notion table

    def __init__(self, email: str = None, password: str = None):
        try:
            if email is not None and password is not None:
                self.pinterest = Pinterest(email=email, password=password)
            else:
                self.__setup()
        except Exception as e:
            self.__setup()

    def __setup(self):
        self.pinterest = Pinterest()

    def __reset_list(self):
        if 0 != len(self.image_info_list):
            self.image_info_list = []
        if 0 != len(self.pin_id_list):
            self.pin_id_list = []
        if 0 != len(self.board_id_list):
            self.board_id_list = []
            self.board_pin_count_list = []

    def search(self, query: str, num_pins: int, scope: str = 'boards'):
        """
        ref: https://github.com/bstoilov/py3-pinterest/blob/master/examples.py#L146
        :param query:
        :param num_pins:
        :param scope:
        :return:
        """
        try:
            self.__setup()  # todo: other handling?
            self.__reset_list()
            search_batch = self._search(
                scope=scope, query=query, page_size=num_pins)

            """
            results = []
            while len(search_batch) > 0 and len(results) < num_pins:
                results += search_batch
                search_batch = self.pinterest.search(scope=scope, query=query)
            self.image_info_list: list = [item['image_cover_hd_url'] for item in results]
            """

            for item in search_batch:
                self.image_info_list.append(item['image_cover_hd_url'])
                # self.pin_id_list.append(item['pin_id'])
                match scope:
                    case 'boards':
                        self.board_id_list.append(item['id'])
                        self.board_pin_count_list.append(item['pin_count'])
                    case _:
                        self.pin_id_list.append(item['id'])
        except Exception as e:
            raise e

    def _search(self, query: str, page_size: int, scope: str = 'boards'):
        try:
            return self.pinterest.search(
                scope=scope, query=query, page_size=page_size)
        except ConnectionError as e:
            logger.error(e)
            self.pinterest = Pinterest()
            return self.pinterest.search(
                scope=scope, query=query, page_size=page_size)

    def get_pin_count(self, board_id: str | int):
        _idx = self.board_id_list.index(str(board_id))
        return self.board_pin_count_list[_idx]

    def get_board_images_from_pin_id(self, pin_id: str, page_size: int = 100):
        self.pin_id: str = get_only_number(text=pin_id)
        board_id = self.get_board_id_from_pin_id(pin_id=self.pin_id)
        return self.get_board_feed_orig_images(
            board_id=board_id, page_size=page_size)

    def get_board_id_from_pin_id(self, pin_id: str):
        pin_info = self.pinterest.load_pin(pin_id=pin_id)
        board_id = pin_info["board"]["id"]
        return board_id

    def get_board_feed_orig_images(self, board_id: str, page_size: int = 100):
        """
        ref: https://github.com/bstoilov/py3-pinterest#list-all-pins-in-board
        :param board_id:
        :param page_size:
        :return:
        """
        self.__setup()  # todo: other handling?
        self.__reset_list()
        self.board_id = board_id

        if page_size > 250:
            result = []
            search_batch = self.pinterest.board_feed(
                board_id=board_id, page_size=250)
            while len(search_batch) > 0 and len(result) < page_size:
                result += search_batch
                search_batch = self.pinterest.board_feed(
                    board_id=board_id, page_size=250)
        else:
            result = self.pinterest.board_feed(
                board_id=board_id, page_size=page_size)
        for item in result:
            with suppress(BaseException):
                self.image_info_list.append(item['images']['orig']['url'])
                self.pin_id_list.append((item['id']))
        return self.image_info_list

    def follow_board(self, board_id=''):
        return self.pinterest.follow_board(board_id=board_id)

    def get_boards(self, username=None):
        return self.pinterest.boards_all(username=username)

    def get_user_boards_batched(self, username=None):
        boards = []
        board_batch = self.pinterest.boards(username=username)
        while len(board_batch) > 0:
            boards += board_batch
            board_batch = self.pinterest.boards(username=username)

        return boards
