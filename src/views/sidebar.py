import io
import logging
import traceback

import streamlit as st

from .pinterest_view import PinterestView, PinterestDemoView

logger = logging.getLogger(__name__)

try:
    from ..services.image_service import ImageService
    from ..services.image_service import SearchImageService
    from ..services.image_service import DownloadImageService
    from ..services.image_services.pinterest_service import PinterestService

    from ..services.csv_service import CsvService
    from ..services.csv_service import get_classification_buffer_data
    from ..services.csv_service import get_regression_buffer_data

    from ..services.notion_service import NotionService
    from .notion_pinterest_view import NotionPinterestView
    from ..services.version_service import VersionService
except ImportError:
    logger.info('check: ImportError')  # todo: fix import error
    from services.image_service import ImageService
    from services.image_service import SearchImageService
    from services.image_service import DownloadImageService

    from services.image_services.pinterest_service import PinterestService

    from services.csv_service import CsvService
    from services.csv_service import get_classification_buffer_data
    from services.csv_service import get_regression_buffer_data

    from services.notion_service import NotionService

    from views.notion_pinterest_view import NotionPinterestView

    from services.version_service import VersionService


class Sidebar:  # todo: refactor

    def __init__(self):
        self.service_dict = {
            'pinterest_service': self.pinterest_service,
            'pinterest_demo_service': self.pinterest_demo_service,
            'csv_service': self.csv_service,
            'notion_service': self.notion_service,
            'notion_pinterest_service': self.notion_pinterest_service,
            'version_service': self.version_service,
        }

    def main(self):
        radio_value = st.sidebar.radio('Sub Page', self.service_dict.keys())
        if radio_value:
            select_service = self.service_dict[radio_value]
            select_service()

    def image_service(self):
        st.title('Image service')

        tab1, tab2, tab3 = st.tabs(['Upload Image Service', 'Search Image Service', 'Download as Zip'])

        with tab1:
            uploaded_file = st.file_uploader('Choose a image file.', type=['jpeg', 'png'])
            try:
                if uploaded_file is not None:
                    image_service = ImageService()
                    image_service.set_image(fp=uploaded_file)
                    st.image(image_service.image, caption='upload image', use_column_width=True)
            except Exception as e:
                logger.error(f'ERROR: {uploaded_file=}')

        with tab2:
            try:
                url: str = st.text_input(label='search url')
                if st.button('Show Images') and 0 != len(url):
                    # ref:https://cafe-mickey.com/python/streamlit-5/
                    search_image = SearchImageService()
                    img_list: list = search_image.get_img_link(url=url)

                    num = 1
                    col = st.columns(num)
                    if 0 != len(img_list):
                        for idx, img_link in enumerate(img_list):
                            with col[idx % num]:
                                st.image(img_list[idx], use_column_width=True)

            except Exception as e:
                logger.error(e)
                logger.error(f'ERROR: Search Image Service')

        with tab3:
            try:
                download_image_service = DownloadImageService(cx=st.secrets['google_custom_search_api']['cx'],
                                                              key=st.secrets['google_custom_search_api']['key'])

                with st.form(key='download_image_service_form'):

                    select_query: str = st.selectbox(label='Select Query', options=download_image_service.query_list)
                    query: str = st.text_input(label='Other Query')
                    num_images: int = st.slider('Num of Images', 0, 100, 25)
                    submitted = st.form_submit_button(label='Setup Download')

                if select_query is not None and num_images is not None and submitted:
                    with st.spinner('Wait for it...'):
                        with io.BytesIO() as buffer:  # ref: https://discuss.streamlit.io/t/download-zipped-json-file/22512/5
                            _query: str = query if 0 != len(query) else select_query
                            zipfile = download_image_service.download_images_as_zipfile(buffer=buffer, query=_query,
                                                                                        num=num_images)
                            buffer.seek(0)

                            if zipfile is not None:
                                st.download_button(label='Download Images as ZipFile',
                                                   data=zipfile,
                                                   file_name='images.zip',
                                                   mime='application/zip')
            except Exception as e:
                logger.error(e)
                traceback.print_exc()

    def pinterest_service(self):
        pinterest_view = PinterestView()
        pinterest_view.main()

    def pinterest_demo_service(self):
        pinterest_demo_view = PinterestDemoView()
        pinterest_demo_view.main()

    def csv_service(self):
        st.title('CSV service')

        tab1, tab2 = st.tabs(['Use Your CSV File', 'Use Temp Data'])

        if 'is_use_tmp_classification_data' not in st.session_state and 'is_use_tmp_regression_data' not in st.session_state:
            st.session_state.is_use_tmp_classification_data = False
            st.session_state.is_use_tmp_regression_data = False

        # todo: session handling, when exists csv in uploaded_file zone
        with tab2:  # temp data tab
            if st.button('Use Classification Data'):
                st.session_state.is_use_tmp_classification_data = True
                st.session_state.is_use_tmp_regression_data = False
            if st.button('Use Regression Data'):
                st.session_state.is_use_tmp_classification_data = False
                st.session_state.is_use_tmp_regression_data = True
        with tab1:  # uploaded_files of csv tab
            uploaded_files = st.file_uploader("Or Your CSV file", type='csv', accept_multiple_files=False)
            if uploaded_files is not None and (
                    st.session_state.is_use_tmp_classification_data or st.session_state.is_use_tmp_regression_data
            ) and st.button('Use Upload CSV File'):
                st.session_state.is_use_tmp_classification_data = False
                st.session_state.is_use_tmp_regression_data = False

        if st.session_state.is_use_tmp_classification_data and st.session_state.is_use_tmp_regression_data:
            raise ValueError
        elif st.session_state.is_use_tmp_classification_data or st.session_state.is_use_tmp_regression_data:
            st.metric(label='Now Select Data', value='Temp Data')
        elif uploaded_files is not None:
            st.metric(label='Now Select Data', value='Upload CSV File')

        try:
            if st.session_state.is_use_tmp_classification_data or st.session_state.is_use_tmp_regression_data or uploaded_files is not None:
                with st.spinner('Wait for it...'):
                    if st.session_state.is_use_tmp_classification_data:
                        csv_service = CsvService(filepath_or_buffer=get_classification_buffer_data())
                    elif st.session_state.is_use_tmp_regression_data:
                        csv_service = CsvService(filepath_or_buffer=get_regression_buffer_data())
                    elif uploaded_files is not None:
                        csv_service = CsvService(filepath_or_buffer=uploaded_files)
                    else:
                        raise ValueError

                tab1, tab2 = st.tabs(['Data Info', 'sklearn Service'])

                with tab1:  # Check Upload CSV File
                    with st.expander(label='Show Data'):
                        st.table(csv_service.df)

                    with st.expander(label='Show Graph of Data', expanded=True):
                        st.line_chart(csv_service.df)

                    with st.expander(label='Show Diff Column'):
                        st.table(csv_service.calc_diff())

                with tab2:  # sklearn Service
                    pass

        except Exception as e:
            logger.error(f'ERROR: {uploaded_files=}')

    def notion_service(self):
        st.title('Notion Service')

        try:
            notion_service = NotionService(access_token=st.secrets['notion_service']['access_token'])
            if st.button('GET'):
                with st.spinner('Wait for it...'):
                    res = notion_service.show_database(database_id=st.secrets['notion_service']['database_id'])
                st.table(res)
                st.json(notion_service.result_dict)
        except Exception as e:
            logger.error(e)
            st.error('access_token error')

    def notion_pinterest_service(self):
        notion_pinterest_view = NotionPinterestView()
        notion_pinterest_view.main()

    def version_service(self):
        st.title('Version Service')
        version_service = VersionService()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label='Python Version', value=version_service.get_python_version())
        with c2:
            st.metric(label='Pip Version', value=version_service.get_pip_version())
        with c3:
            st.metric(label='Streamlit Version',
                      value=version_service.get_library_version(library_name='streamlit'))
        st.download_button(label='Download requirements.txt',
                           data=version_service.get_pip_list(format='freeze'),
                           file_name='requirements.txt',
                           mime='text/txt')
        with st.spinner('Wait for it...'):
            pip_list = version_service.get_pip_list(format='json')
        with st.expander('Pip List', expanded=True):
            st.table(pip_list)
