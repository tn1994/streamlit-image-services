import uuid
from abc import ABC, abstractmethod

import pandas as pd
import streamlit as st


class BaseView(ABC):
    """
    ref: https://twitter.com/testdrivenio/status/1579147179934904320?s=20&t=URxZZzdX3Icdc2gftdIXAw
    """

    @classmethod
    @property
    @abstractmethod
    def title(cls):
        raise NotImplementedError

    def main(self):
        raise NotImplementedError

    @staticmethod
    def download_df_as_csv(df: pd.DataFrame, file_name='filelist.csv'):
        st.download_button(
            key=uuid.uuid1(),
            label='Download csv',
            data=df.to_csv(
                index=False),
            file_name=file_name,
            mime='text/csv')


def spinner_wrapper(func):
    def wrapper(*args, **kwargs):
        with st.spinner('Wait for it...'):
            func(*args, **kwargs)

    return wrapper
