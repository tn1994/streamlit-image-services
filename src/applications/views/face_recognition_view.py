import logging

import pandas as pd
import streamlit as st

from ..services.torch_services.datasets.dataset import generate_face_recognition

logger = logging.getLogger(__name__)


def show_face_recognition(image_info_list: list) -> list:
    if isinstance(image_info_list, pd.Series):
        image_info_list: list = image_info_list.tolist()

    if 0 != len(image_info_list):
        _face_recognition_list: list = []
        for idx, img_link in enumerate(image_info_list):
            with st.expander(label='Show Face Recognition', expanded=True):
                col1, col2 = st.columns(2)  # must in expander
                with col1:
                    st.image(image_info_list[idx], use_column_width=True)
                with col2:
                    image = generate_face_recognition(
                        path=image_info_list[idx], is_get_pil_image=True)
                    if image is None:
                        image = image_info_list[idx]
                    else:
                        _face_recognition_list.append(image_info_list[idx])
                    st.image(image, use_column_width=True)
        return _face_recognition_list
    return []
