import io
import os
import shutil
import requests

import cv2
import torch
import torchvision
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


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
    df = None

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


def setup_data(df: pd.DataFrame):
    import face_recognition
    # !pip3 install face_recognition
    # !rm - rf. / face_recognition /.ipynb_checkpoints
    # !ls - a
    # face_recognition

    # '901494050386870050', '901494050386841932'
    list_a = df.query('label == 901494050386870050')['link'].tolist()
    list_b = df.query('label == 901494050386841932')['link'].tolist()

    # 上に合わせて書き換えてください！！
    # data_list = [MyDataset.image_link_list]
    data_list = [list_a, list_b]

    # クラス名を記載してください！
    name_list = ['sayashi_riho', 'oda_sakura']
    for name, l in zip(name_list, data_list):
        count = 1
        dir_name: str = f'face_recognition/{name}'
        os.makedirs(dir_name, exist_ok=True)
        for idx, path in enumerate(l):
            print(path)
            image = imread_web(path)  # cv2
            if image is None:
                print('Image is None: ', path)
                continue
            faces = face_recognition.face_locations(image)
            if 0 != len(faces):
                top, right, bottom, left = faces[0]
                print(
                    "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left,
                                                                                                          bottom,
                                                                                                          right))
                face_image = image[top:bottom, left:right]
                cv2.imwrite(f"./{dir_name}/{count}.png", face_image)
                count += 1
            else:
                print('Image is None: ', path)


def get_dataset():
    # ref: https://qiita.com/ryryry/items/b1da4855504dcd3f9d98

    data_folder = './face_recognition/'

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

    data = torchvision.datasets.ImageFolder(root=data_folder, transform=transform_dict[phase])

    train_ratio = 0.8

    train_size = int(train_ratio * len(data))
    # int()で整数に。
    val_size = len(data) - train_size
    data_size = {"train": train_size, "val": val_size}
    #          =>{"train": 112,       "val": 28}
    train_dataset, valid_dataset = torch.utils.data.random_split(data, [train_size, val_size])
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


def get_image_for_save(image_link: str):
    res = requests.get(image_link, stream=True)
    if res.status_code == 200:
        filename: str = os.path.basename(image_link)
        save_image(filename=filename, obj=res.raw)


def save_image(filename: str, obj, is_in_dir: bool = True):
    is_in_dir = False
    if is_in_dir:
        dir_name: str = ''
        os.makedirs(dir_name, exist_ok=True)
        path: str = os.path.join(dir_name, filename)
    else:
        path: str = filename

    with open(path, 'wb') as f:
        obj.decode_content = True
        shutil.copyfileobj(obj, f)


def imread_web(url):
    """
    ref: https://ensekitt.hatenablog.com/entry/2018/06/25/200000
    """
    import tempfile
    # 画像をリクエストする
    res = requests.get(url)
    img = None
    # Tempfileを作成して即読み込む
    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img
