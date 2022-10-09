import logging

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

            with st.expander(label='Show Pins', expanded=True):
                num = 3
                col = st.columns(num)
                if 0 != len(pinterest_service.image_info_list):
                    for idx, img_link in enumerate(pinterest_service.image_info_list):
                        with col[idx % num]:
                            st.image(pinterest_service.image_info_list[idx], use_column_width=True)


class PinterestDemoView:
    title: str = 'Pinterest Demo Service'

    def main(self):
        st.title(self.title)

        pinterest_service = PinterestService()

        with st.form(key='pinterest_service_form'):
            # select_query: str = st.selectbox(label='Select Query', options=pinterest_service.query_list)
            pin_id: str = st.text_input(label='Pin ID', value='901494050386847677')
            board_id: str = st.text_input(label='Board ID')
            num_pins: int = st.slider('Num of Images', 0, 100, 25)
            submitted = st.form_submit_button(label='Search')

        logger.info(f'{pin_id=}, {board_id=}, {num_pins=}')

        if (0 != len(pin_id) or 0 != len(board_id)) and num_pins is not None and submitted:
            with st.spinner('Wait for it...'):
                if 0 != len(pin_id):
                    image_list = pinterest_service.get_board_images_from_pin_id(pin_id=pin_id,
                                                                                page_size=num_pins)
                elif 0 != len(board_id):
                    image_list = pinterest_service.get_board_feed_orig_images(board_id=board_id,
                                                                              page_size=num_pins)
            self._download(data=pinterest_service.image_info_list)
            st.json(pinterest_service.pin_id_list)
            with st.expander(label='Show Pins', expanded=True):
                num = 3
                col = st.columns(num)
                if 0 != len(image_list):
                    for idx, img_link in enumerate(image_list):
                        with col[idx % num]:
                            st.image(image_list[idx], use_column_width=True)

    def _download(self, data):
        if isinstance(data, list):
            import pandas as pd
            data = pd.DataFrame({'filelist': data}).to_csv(index=False)
        st.download_button(label='Download csv', data=data, file_name='filelist.csv', mime='text/csv')
