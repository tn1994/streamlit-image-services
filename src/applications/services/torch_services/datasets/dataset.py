import io
import os
import shutil
import logging
import requests
from typing import Optional

import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import face_recognition
from torchvision import transforms
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    filename_list = [
        'a', 'b', 'c', 'd', 'e', 'f'
    ]
    image_link_list = [
        'https://i.pinimg.com/550x/a0/a9/7b/a0a97b1c9e5ecdb8ba6b6c3fd076f0da.jpg',
        'https://i.pinimg.com/550x/33/0b/fe/330bfebf47fa99c0896e4996338b4f7e.jpg',
        'https://i.pinimg.com/550x/e1/20/ed/e120ed80ec5a01d6a9857accec731482.jpg',

        'https://i.pinimg.com/originals/40/bb/ad/40bbadb13def4ca573a5b017553e1ce0.png',
        'https://i.pinimg.com/originals/5e/82/92/5e8292682b884bae1fc9671d8444261d.jpg',
        'https://i.pinimg.com/originals/6f/b0/9e/6fb09e85d0afe9569795edb10298553b.jpg'
    ]

    transformer = None
    df: pd.DataFrame = None

    def __init__(self, df=None):
        if df is not None:
            self.df = df.copy()
            self.filename_list = self.df['label'].tolist()
            self.image_link_list = self.df['link'].tolist()
            self.num_class: int = len(self.df['label'].unique().tolist())

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        """
        Notes:
        returnに関して、ログ蓄積として「ファイル名と正解値、予測値」を残すため、
        本ファンクションでは「ファイル名」も返却している。
        :param index:
        :return:
        """
        filename: str = self.filename_list[index]
        image_link: str = self.image_link_list[index]

        # 画像の読込
        image = get_image(image_link=image_link)
        image = image.convert('RGB')

        # 画像への前処理適用
        if self.transformer is not None:
            image = self.transformer(image)
        else:
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # input_tensor = preprocess(input_image)
            # image = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            image = preprocess(image)

        # 差分画像の正解値取得
        # correct_value: float = self._get_correct_value(filename=filename)

        correct_value = 500

        result_dict: dict = {
            'image_id': self.filename_list[index],
            'image': image
        }
        if self.df is not None:
            class_index = self.df['label'].unique().tolist().index(self.filename_list[index])
            result_dict['image_class_idx'] = class_index

        return result_dict


def setup_data(df: pd.DataFrame, is_face_recognition: bool, timestamp: str):
    try:
        df: pd.DataFrame = df.copy()
        # '901494050386870050', '901494050386841932'
        # list_a = df.query('label == 901494050386870050')['link'].tolist()
        # list_b = df.query('label == 901494050386841932')['link'].tolist()

        # 上に合わせて書き換えてください！！
        # data_list = [MyDataset.image_link_list]
        # data_list = [list_a, list_b]

        label_list: list = df['label'].unique().tolist()

        data_list: list = []

        logger.info(f'{label_list=}')
        for label in label_list:
            data_list.append(df.query(f'label == {label}')['link'].tolist())

        logger.info(f'{label_list=}, {data_list=}')
        # クラス名を記載してください！
        # name_list = ['sayashi_riho', 'oda_sakura']
        name_list: list = label_list
        logger.info(f'{label_list=}, {name_list=}')

        if 1 != len(label_list) != len(name_list):
            raise ValueError

        for name, l in zip(name_list, data_list):
            count = 1
            for idx, path in enumerate(l):
                if is_face_recognition:
                    face_image = generate_face_recognition(path)
                    if face_image is not None:
                        dir_name: str = f'{timestamp}/face_recognition/{name}'
                        os.makedirs(dir_name, exist_ok=True)
                        _path: str = f'./{dir_name}/{count}.png'
                        cv2.imwrite(_path, face_image)
                        logger.info(f'Check it: {_path}')
                        count += 1
                    else:
                        print('Image is None: ', path)
                else:
                    dir_name: str = f'{timestamp}/original/{name}'
                    os.makedirs(dir_name, exist_ok=True)
                    get_image_for_save(image_link=path, dir_name=f'./{dir_name}')
    except Exception as e:
        raise e


def generate_face_recognition(path: str, is_get_pil_image: bool = False):
    image = imread_web(path)  # cv2
    if image is None:
        logger.info('Image is None: ', path)
    else:
        faces = face_recognition.face_locations(image)
        if 0 != len(faces):
            top, right, bottom, left = faces[0]
            logger.info(f"A face is located at pixel location "
                        f"Top: {top}, Left: {left}, "
                        f"Bottom: {bottom}, Right: {right}")
            face_image = image[top:bottom, left:right]
            if is_get_pil_image:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            return face_image
    return None


def get_dataset(data_folder='./face_recognition/'):
    # ref: https://qiita.com/ryryry/items/b1da4855504dcd3f9d98

    transform_dict = {
        'train': transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'test': transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}

    phase = 'train'

    dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=transform_dict[phase])
    return dataset


def split_train_valid_dataset(dataset):
    train_ratio = 0.8

    train_size = int(train_ratio * len(dataset))
    # int()で整数に。
    val_size = len(dataset) - train_size
    data_size = {"train": train_size, "val": val_size}
    #          =>{"train": 112,       "val": 28}
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, valid_dataset


def get_image(image_link: str) -> Image:
    """
    ref: https://r17n.page/2019/08/01/python-requests-download-image/
    """
    res = requests.get(image_link)
    if res.status_code == 200:
        filename: str = os.path.basename(image_link)
        # img = r.raw.read()
        img = Image.open(io.BytesIO(res.content))
        # save_image(filename=filename, obj=r.content)
        return img


def get_image_for_save(image_link: str, dir_name: str):
    res = requests.get(image_link, stream=True)
    if res.status_code == 200:
        filename: str = os.path.basename(image_link)
        save_image(filename=filename, obj=res.raw, dir_name=dir_name)


def save_image(filename: str, obj, dir_name: str = None):
    if dir_name is not None:
        os.makedirs(dir_name, exist_ok=True)
        path: str = os.path.join(dir_name, filename)
    else:
        path: str = filename

    with open(path, 'wb') as f:
        obj.decode_content = True
        shutil.copyfileobj(obj, f)

    logger.info(f'save_image: {path}')


def imread_web(url):
    """
    ref: https://qiita.com/derodero24/items/f22c22b22451609908ee
    """
    res = requests.get(url)
    pil_image = Image.open(io.BytesIO(res.content))
    new_image = np.array(pil_image, dtype=np.uint8)
    img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    return img
