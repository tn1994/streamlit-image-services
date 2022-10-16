import logging

import streamlit as st

from applications.views.sidebar import Sidebar
from applications.views.about_me import AboutMe
from applications.services.utils.analysis_memory import get_memory_state_percent

logger = logging.getLogger(__name__)


class APP:
    env: str = st.secrets["env"]
    hashed_text: str = st.secrets['hashed_text']

    if 'is_authorization' not in st.session_state:
        st.session_state.is_authorization = False

    def __init__(self):
        if self.env not in ('prod', 'develop'):
            raise ValueError
        st.set_page_config(
            page_title='tn1994/streamlit-image-services',
            layout='wide'
        )

    def main(self):
        self._sidebar()

    def _top_page(self):
        self._title()
        AboutMe.main()

    @staticmethod
    def _title():
        st.title('streamlit-demo')

    @staticmethod
    def _sidebar():
        sidebar = Sidebar()
        sidebar.main()


def main():
    app = APP()
    app.main()


if __name__ == '__main__':
    main()
