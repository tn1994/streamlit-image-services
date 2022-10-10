import uuid
import logging

import pandas as pd
import streamlit as st

from ..services.image_services.pinterest_service import PinterestService
from ..services.torch_services.datasets.dataset import generate_face_recognition

from ..views.face_recognition_view import show_face_recognition

logger = logging.getLogger(__name__)


class PinterestView:
    title: str = 'Pinterest Service'

    def main(self):
        st.title(self.title)

        pinterest_service = PinterestService()

        with st.form(key='pinterest_service_form'):
            select_query: str = st.selectbox(label='Select Query', options=pinterest_service.query_list)
            query: str = st.text_input(label='Other Query')
            num_pins: int = st.slider('Num of Images', 0, 100, 25)
            submitted = st.form_submit_button(label='Search')

        if 0 != len(select_query) and num_pins is not None and submitted:
            with st.spinner('Wait for it...'):
                _query: str = query if 0 != len(query) else select_query
                pinterest_service.search(query=_query, num_pins=num_pins)

            st.table(pinterest_service.pin_id_list)
            st.table(pinterest_service.image_info_list)

            with st.expander(label='Show Pins', expanded=True):
                num = 3
                col = st.columns(num)
                if 0 != len(pinterest_service.image_info_list):
                    for idx, img_link in enumerate(pinterest_service.image_info_list):
                        with col[idx % num]:
                            st.image(pinterest_service.image_info_list[idx], use_column_width=True)


class PinterestDemoView:
    title: str = 'Pinterest Board Service'

    def main(self):
        st.title(self.title)

        pinterest_service = PinterestService()

        with st.form(key='pinterest_service_form'):
            # select_query: str = st.selectbox(label='Select Query', options=pinterest_service.query_list)
            pin_id: str = st.text_input(label='Pin ID of Board', value='901494050386847677')
            board_id: str = st.text_input(label='Board ID')
            num_pins: int = st.slider('Num of Images', 1, 300, 100)
            col1, col2 = st.columns(2)
            with col1:
                submitted_search = st.form_submit_button(label='Search')
            with col2:
                submitted_face_recognition = st.form_submit_button(label='Face Recognition')

        if (0 != len(pin_id) or 0 != len(board_id)) and num_pins is not None and (
                submitted_search or submitted_face_recognition):
            with st.spinner('Wait for it...'):
                if 0 != len(pin_id):
                    pinterest_service.get_board_images_from_pin_id(pin_id=pin_id,
                                                                   page_size=num_pins)
                    self._download(data=pinterest_service.image_info_list, label=pin_id)
                elif 0 != len(board_id):
                    pinterest_service.get_board_feed_orig_images(board_id=board_id,
                                                                 page_size=num_pins)
                    self._download(data=pinterest_service.image_info_list, label=board_id)
                with st.expander(label='Show Pins ID', expanded=False):
                    st.table(pinterest_service.pin_id_list)
                with st.expander(label='Show Pins Link', expanded=False):
                    st.table(pinterest_service.image_info_list)

                if submitted_face_recognition:
                    self._show_face_recognition(image_info_list=pinterest_service.image_info_list,
                                                label_id=pin_id if 0 != len(pin_id) else board_id)
                else:
                    with st.expander(label='Show Pins', expanded=True):
                        num = 3
                        col = st.columns(num)
                        if 0 != len(pinterest_service.image_info_list):
                            for idx, img_link in enumerate(pinterest_service.image_info_list):
                                with col[idx % num]:
                                    st.image(pinterest_service.image_info_list[idx], use_column_width=True)

    def _download(self, data, label: str):
        if isinstance(data, list):
            data = pd.DataFrame({'link': data,
                                 'label': [label for _ in range(len(data))]}).to_csv(index=False)
        st.download_button(key=uuid.uuid1(), label='Download csv', data=data, file_name='filelist.csv', mime='text/csv')

    def _show_face_recognition(self, image_info_list: list, label_id: str):
        if 0 != len(image_info_list):
            _face_recognition_list: list = show_face_recognition(image_info_list=image_info_list)
            self._download(data=_face_recognition_list, label=label_id)
