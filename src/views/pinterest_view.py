import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

try:
    from ..services.image_services.pinterest_service import PinterestService
except ImportError as e:
    logger.error(e)
    from services.image_services.pinterest_service import PinterestService


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
            num_pins: int = st.slider('Num of Images', 0, 300, 100)
            submitted = st.form_submit_button(label='Search')

        logger.info(f'{pin_id=}, {board_id=}, {num_pins=}')

        if (0 != len(pin_id) or 0 != len(board_id)) and num_pins is not None and submitted:
            with st.spinner('Wait for it...'):
                if 0 != len(pin_id):
                    pinterest_service.get_board_images_from_pin_id(pin_id=pin_id,
                                                                   page_size=num_pins)
                elif 0 != len(board_id):
                    pinterest_service.get_board_feed_orig_images(board_id=board_id,
                                                                 page_size=num_pins)
        if 0 != len(pinterest_service.image_info_list):
            image_list: list = pinterest_service.image_info_list
            self._download(data=image_list)
            with st.expander(label='Show Pins ID', expanded=False):
                st.table(pinterest_service.pin_id_list)
            with st.expander(label='Show Pins Link', expanded=False):
                st.table(image_list)
            with st.expander(label='Show Pins', expanded=True):
                num = 3
                col = st.columns(num)
                if 0 != len(image_list):
                    for idx, img_link in enumerate(image_list):
                        with col[idx % num]:
                            st.image(image_list[idx], use_column_width=True)

    def _download(self, data):
        with st.form(key='pinterest_csv_download_service_form'):
            st.write('Download CSV Form')
            label = st.text_input(label='Set Label: Add Column')
            if isinstance(data, list):
                data = pd.DataFrame({'filelist': data,
                                     'label': [label for _ in range(len(data))]}).to_csv(index=False)
            submitted = st.form_submit_button(label='Setup')
        if submitted:
            st.download_button(label='Download csv', data=data, file_name='filelist.csv', mime='text/csv')
