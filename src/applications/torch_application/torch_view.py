import logging

import streamlit as st

from ..services.csv_service import CsvService
from ..services.torch_services.datasets.dataset import generate_face_recognition

logger = logging.getLogger(__name__)


class TorchView:
    title: str = 'Pinterest Service'

    def main(self):
        st.title(self.title)
        uploaded_files = st.file_uploader("Or Your CSV file", type='csv', accept_multiple_files=False)

        try:
            if uploaded_files is not None:
                with st.spinner('Wait for it...'):
                    if uploaded_files is not None:
                        csv_service = CsvService(filepath_or_buffer=uploaded_files)
                    else:
                        raise ValueError
                with st.expander(label='Show Data'):
                    st.table(csv_service.df)
                if st.button(label='Train'):
                    pass
        except Exception as e:
            raise e
