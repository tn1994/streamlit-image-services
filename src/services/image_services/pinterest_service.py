import logging

from py3pin.Pinterest import Pinterest

logger = logging.getLogger(__name__)


class PinterestService:
    """
    ref: https://github.com/bstoilov/py3-pinterest
    """

    image_info_list: list = []
    pin_id_list: list = []

    query_list: list = ['鞘師里保', '小田さくら', '佐藤優樹', 'モーニング娘。']

    def __init__(self, email: str = None, password: str = None):
        try:
            if email is not None and password is not None:
                self.pinterest = Pinterest(email=email, password=password)
            else:
                self.pinterest = Pinterest()
        except Exception as e:
            self.pinterest = Pinterest()

    def search(self, query: str, num_pins: int, scope: str = 'boards'):
        """
        ref: https://github.com/bstoilov/py3-pinterest/blob/master/examples.py#L146
        :param query:
        :param num_pins:
        :param scope:
        :return:
        """
        try:
            if 0 != len(self.image_info_list):
                self.image_info_list = []
            if 0 != len(self.pin_id_list):
                self.pin_id_list = []
            search_batch = self._search(scope=scope, query=query, page_size=num_pins)
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
                self.pin_id_list.append(item['id'])
        except Exception as e:
            raise e

    def _search(self, query: str, page_size: int, scope: str = 'boards'):
        return self.pinterest.search(scope=scope, query=query, page_size=page_size)

    def get_board_images_from_pin_id(self, pin_id: str, page_size: int = 100):
        board_id = self.get_board_id_from_pin_id(pin_id=pin_id)
        return self.get_board_feed_orig_images(board_id=board_id, page_size=page_size)

    def get_board_id_from_pin_id(self, pin_id: str):
        # pin_id: str = '901494050386847677'  # sayashi riho
        # pin_id: str = '901494050386841932'  # oda sakura

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
        if 0 != len(self.image_info_list):
            self.image_info_list = []
        if 0 != len(self.pin_id_list):
            self.pin_id_list = []
        result = self.pinterest.board_feed(board_id=board_id, page_size=page_size)
        for item in result:
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
