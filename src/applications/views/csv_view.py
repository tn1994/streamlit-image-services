import uuid
import logging

import streamlit as st

from .base_view import BaseView
from ..services.csv_service import CsvService
from ..services.csv_service import concat_df

logger = logging.getLogger(__name__)


class CsvView(BaseView):
    title: str = 'CSV service'

    def main(self):
        st.title(self.title)

        self.viewer()

        self.concat()

    def viewer(self):
        with st.expander(label='Viewew', expanded=True):
            uploaded_files = st.file_uploader(
                "Or Your CSV file", type='csv', accept_multiple_files=False)

            try:
                if uploaded_files is not None:
                    with st.spinner('Wait for it...'):
                        if uploaded_files is not None:
                            csv_service = CsvService(
                                filepath_or_buffer=uploaded_files)
                        else:
                            raise ValueError
                    with st.expander(label='Show Data'):
                        st.table(csv_service.df)
                    if st.button(label='Train'):
                        pass

            except Exception as e:
                logger.error(f'ERROR: {uploaded_files=}')

    def concat(self):
        with st.expander(label='Concut', expanded=True):
            uploaded_files = st.file_uploader(
                "Or Your CSV file", type='csv', accept_multiple_files=True)

            try:
                if uploaded_files:
                    with st.spinner('Wait for it...'):
                        logger.info(uploaded_files)
                        df_in_list = []
                        for num in range(len(uploaded_files)):
                            df_in_list.append(
                                CsvService(
                                    filepath_or_buffer=uploaded_files[num]).df)
                        data = concat_df(df_in_list=df_in_list)
                        st.table(data)
                    self.download_df_as_csv(df=data)
            except Exception as e:
                logger.error(e)
