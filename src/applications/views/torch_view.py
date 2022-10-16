import io
import uuid
import logging

import pandas as pd
import streamlit as st

from .base_view import spinner_wrapper

from ..services.csv_service import CsvService
from ..services.torch_services.utils import is_cuda_available
from ..services.torch_services.torch_sevice import TorchService
from ..services.utils.convert_zipfile import ZipFile
from ..services.torch_services.models.model import model_name_list

from ..views.face_recognition_view import show_face_recognition

logger = logging.getLogger(__name__)


class TorchView:
    title: str = 'Torch Service'

    is_cuda_available: bool = is_cuda_available()

    key_formm_of_fit_model: str = str(uuid.uuid1())

    def main(self):
        st.title(self.title)
        st.metric(label='is_cuda_available', value=self.is_cuda_available)

        st.markdown('## 1. Setup Data')
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
                self.anaysis_view(df=csv_service.df)

                st.markdown('## 2. Check Data')
                self._show_face_recognition(df=csv_service.df)

                st.markdown('## 3. Setup Train')
                self.fit_model(df=csv_service.df)

        except Exception as e:
            raise e

    def anaysis_view(self, df: pd.DataFrame):
        df: pd.DataFrame = df.copy()
        label_list = df['label'].unique()
        _value_counts = df['label'].value_counts()
        num_link = [_value_counts[label] for label in label_list]
        st.table(
            pd.DataFrame({
                'label': df['label'].unique(),
                'num_link': num_link
            })
        )

    @spinner_wrapper
    def _show_face_recognition(self, df: pd.DataFrame):
        df: pd.DataFrame = df.copy()
        select_label = st.selectbox(
            label='Select Label',
            options=df['label'].unique())
        num_images: int = st.slider('Num of Images', 1, df.query(
            f'label == {select_label}').__len__(), 5)
        is_show: bool = st.selectbox(
            label='Is Show Face Recognition', options=[
                False, True])

        if num_images and is_show and select_label:
            image_info_list: list = df.query(f'label == {select_label}')[
                'link'].tolist()
            _image_info_list = image_info_list[:num_images]
            show_face_recognition(image_info_list=_image_info_list)

    def fit_model(self, df: pd.DataFrame):
        df: pd.DataFrame = df.copy()

        with st.form(key=self.key_formm_of_fit_model):
            use_images: str = st.selectbox(
                label='Use Image Is', options=[
                    'Original', 'Face Recognition'])
            select_model: str = st.selectbox(
                label='Use Model Is', options=model_name_list)
            # query: str = st.text_input(label='Other Query')
            num_epochs: int = st.slider('Num of Epochs', 1, 100, 25)
            download_after_train: str = st.selectbox(
                label='Download After Train Is', options=[
                    'Nothing', 'All Data and Model', 'Model Only'])
            submitted = st.form_submit_button(label='Train')

        # if use_images is not None and select_model is not None and
        # download_after_train is not None and submitted:
        if all([use_images,
                select_model,
                download_after_train,
                num_epochs]) and submitted:
            match self.is_cuda_available:
                case False:
                    with st.spinner('Trining Model Now...'):
                        torch_service = TorchService()
                        _is_face_recognition = True if use_images == 'Face Recognition' else False
                        torch_service.main(
                            df=df,
                            model_name=select_model,
                            num_epochs=num_epochs,
                            is_face_recognition=_is_face_recognition)
                case False:
                    st.error('Cuda is not available. please change server.')
                case _:
                    raise

            self._download_data(download_after_train=download_after_train,
                                torch_service=torch_service)

    @spinner_wrapper
    def _download_data(self, download_after_train: str, torch_service):
        after_train_message = '## 4. Download' if download_after_train != 'Nothing' else '## 4. Fin'
        st.markdown(after_train_message)
        match download_after_train:
            case 'Model Only':
                with open(f'./{torch_service.timestamp}/model.pth', 'rb') as fp:
                    st.download_button(label='Download Only Model',
                                       data=fp,
                                       file_name='model.pth',
                                       mime='application/zip')
            case 'All Data and Model':
                with io.BytesIO() as buffer:
                    zipfile = ZipFile()
                    zipfile.archive_dir(
                        dir_name=f'./{torch_service.timestamp}', buffer=buffer)
                    st.download_button(label='Download All Data',
                                       data=zipfile.buffer,
                                       file_name='data.zip',
                                       mime='application/zip')
