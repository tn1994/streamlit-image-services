import uuid
import logging

import pandas as pd
import streamlit as st

from .base_view import spinner_wrapper
from ..views.face_recognition_view import show_face_recognition
from ..services.image_services.pinterest_service import PinterestService

logger = logging.getLogger(__name__)


class PinterestBaseView:

    @staticmethod
    @spinner_wrapper
    def show_pins_images(image_info_list: list):
        with st.expander(label='Show Pins', expanded=True):
            num: int = st.selectbox(label='Num of Columns', options=[3, 2, 1])
            col = st.columns(num)
            if 0 != len(image_info_list):
                for idx, img_link in enumerate(image_info_list):
                    with col[idx % num]:
                        st.image(image_info_list[idx], use_column_width=True)
            else:
                st.warning('No Images')


class PinterestView(PinterestBaseView):
    title: str = 'Pinterest Service'

    key_1: str = str(uuid.uuid1())
    key_2: str = str(uuid.uuid1())

    pinterest_service = PinterestService()

    def main(self):
        st.title(self.title)

        self.options_sidebar()

        self.show_board_images()

    def options_sidebar(self):
        st.sidebar.write('Page Options')
        radio_value = st.sidebar.radio(
            label='Query Type', options=[
                'Select', 'Query'])

        match radio_value:
            case 'Select':
                _category: str = st.sidebar.selectbox(
                    label='Group Name', options=self.pinterest_service.query_category_dict.keys())
                _group_name: str = st.sidebar.selectbox(
                    label='Group Name',
                    options=self.pinterest_service.query_category_dict[_category])
                query: str = st.sidebar.selectbox(
                    label='Select Query',
                    options=self.pinterest_service.query_category_dict[_category][_group_name])
                col = st.columns(3)
                col[0].metric(label='Category', value=_category)
                col[1].metric(label='Group', value=_group_name)
                col[2].metric(label='Query', value=query)
            case 'Query':
                query: str = st.text_input(label='Other Query')
            case _:
                raise

        """
        with st.sidebar.form(key=self.key_1):
            num_boards: int = st.slider('Num of Boards', 1, 200, 100)
            submitted = st.form_submit_button(label='Search Boards')

        if 0 != len(query) and num_boards is not None and submitted:
            with st.spinner('Wait for it...'):
                self.pinterest_service.search(
                    query=query, num_pins=num_boards, scope='boards')
                    """

        with st.spinner('Wait for it...'):
            self.pinterest_service.search(
                query=query, num_pins=100, scope='boards')

    def show_board_images(self):
        if 0 != len(self.pinterest_service.board_id_list):
            board_id_list = self.pinterest_service.board_id_list
            select_board_id: str = st.selectbox(
                label='Select Board ID', options=board_id_list)
            with st.form(key=self.key_2):
                _max_num_images = self.pinterest_service.get_pin_count(
                    board_id=select_board_id)
                num_pins: int = st.slider(
                    'Num of Images', 1, self.pinterest_service.get_pin_count(
                        board_id=select_board_id), 100 if _max_num_images > 100 else _max_num_images)
                submitted = st.form_submit_button(label='Show Images')

            if 0 != len(
                    select_board_id) and num_pins is not None and submitted:
                pinterest_service = PinterestService()
                pinterest_service.get_board_feed_orig_images(
                    board_id=select_board_id, page_size=num_pins)
                self.show_pins_images(
                    image_info_list=pinterest_service.image_info_list)
            else:
                pinterest_service = PinterestService()
                pinterest_service.get_board_feed_orig_images(
                    board_id=select_board_id, page_size=100)
                self.show_pins_images(
                    image_info_list=pinterest_service.image_info_list)


class PinterestBoardView(PinterestBaseView):
    title: str = 'Pinterest Board Service'

    def main(self):
        st.title(self.title)

        pinterest_service = PinterestService()

        with st.form(key='pinterest_service_form'):
            pin_id: str = st.text_input(
                label='Pin ID of Board',
                value='901494050386847677')
            board_id: str = st.text_input(label='Board ID')
            num_pins: int = st.slider('Num of Images', 1, 300, 100)
            col1, col2 = st.columns(2)
            with col1:
                submitted_search = st.form_submit_button(label='Search')
            with col2:
                submitted_face_recognition = st.form_submit_button(
                    label='Face Recognition')

        if (0 != len(pin_id) or 0 != len(board_id)) and num_pins is not None and (
                submitted_search or submitted_face_recognition):
            with st.spinner('Wait for it...'):
                if 0 != len(pin_id):
                    pinterest_service.get_board_images_from_pin_id(
                        pin_id=pin_id, page_size=num_pins)
                    self._download(
                        data=pinterest_service.image_info_list,
                        label=pin_id)
                elif 0 != len(board_id):
                    pinterest_service.get_board_feed_orig_images(
                        board_id=board_id, page_size=num_pins)
                    self._download(
                        data=pinterest_service.image_info_list,
                        label=board_id)
                st.table([{'pin_id': pinterest_service.pin_id,
                           'board_id': pinterest_service.board_id}])
                with st.expander(label='Show Pins ID', expanded=False):
                    st.table(pinterest_service.pin_id_list)
                with st.expander(label='Show Pins Link', expanded=False):
                    st.table(pinterest_service.image_info_list)

                if submitted_face_recognition:
                    self._show_face_recognition(
                        image_info_list=pinterest_service.image_info_list,
                        label_id=pin_id if 0 != len(pin_id) else board_id)
                else:
                    self.show_pins_images(
                        image_info_list=pinterest_service.image_info_list)

    @staticmethod
    def _download(data, label: str, button_label: str = 'Download csv'):
        if isinstance(data, list):
            data = pd.DataFrame(
                {'link': data, 'label': [label for _ in range(len(data))]}).to_csv(index=False)
        st.download_button(
            key=uuid.uuid1(),
            label=button_label,
            data=data,
            file_name='filelist.csv',
            mime='text/csv')

    def _show_face_recognition(self, image_info_list: list, label_id: str):
        if 0 != len(image_info_list):
            _face_recognition_list: list = show_face_recognition(
                image_info_list=image_info_list)
            self._download(
                data=_face_recognition_list,
                label=label_id,
                button_label='Download Face Recognition csv')
