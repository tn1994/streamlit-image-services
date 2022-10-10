import os
import copy
import shutil
import random
import requests
import logging
import datetime

import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split

from .datasets.dataset import imread_web
from .datasets.dataset import MyDataset, setup_data, get_dataset
from .models.model import get_model
from .datasets.dataset import split_train_valid_dataset

logger = logging.getLogger(__name__)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

N_WORKERS = 0


class TorchService:
    class_to_idx = {}

    df: pd.DataFrame = None

    model = None

    # https://note.nkmk.me/python-datetime-usage/
    timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    def demo_data(self, df: pd.DataFrame):
        # ref: https://stackoverflow.com/questions/54797508/how-to-generate-a-train-test-split-based-on-a-group-id
        # splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 42)
        splitter = GroupKFold(n_splits=2)
        split = splitter.split(df, groups=df['label'])
        train_inds, test_inds = next(split)

        train = df.iloc[train_inds]
        test = df.iloc[test_inds]

        train, test = train_test_split(df, test_size=0.4, random_state=42)

        train['label'].value_counts()

    def train(self, train_dataset, valid_dataset, num_epochs: int):
        """
        ref: https://venoda.hatenablog.com/entry/2020/10/18/014516
        :return:
        """
        if not torch.cuda.is_available():
            raise
        # バッチサイズの指定
        batch_size = 64

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=N_WORKERS,
            shuffle=True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=32, shuffle=False)

        # 辞書にまとめる
        dataloaders_dict = {
            'train': train_dataloader,
            'valid': valid_dataloader
        }

        # setup fine tuning
        # ref: https://tzmi.hatenablog.com/entry/2020/01/27/001036
        self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features,
                                              out_features=len(self.class_to_idx.keys()))

        optimizer = optim.RAdam(params=self.model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-------------')

            for phase in ('train', 'valid'):
                if phase == 'train':
                    # 学習モードに設定
                    self.model.to('cuda')
                    self.model.train()
                else:
                    # 訓練モードに設定
                    self.model.eval()

                # epochの損失和
                epoch_loss = 0.0
                # epochの正解数
                epoch_corrects = 0

                for step, data in enumerate(dataloaders_dict[phase]):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()

                    # optimizerを初期化
                    optimizer.zero_grad()

                    # 学習時のみ勾配を計算させる設定にする
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = self.model(inputs)

                        # 損失を計算
                        loss = criterion(outputs, labels)

                        # ラベルを予測
                        _, preds = torch.max(outputs, 1)

                        # 訓練時は逆伝搬の計算
                        if phase == 'train':
                            # 逆伝搬の計算
                            loss.backward()

                            # パラメータ更新
                            optimizer.step()

                        # イテレーション結果の計算
                        # lossの合計を更新
                        # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                        # データ数を掛けることで平均から合計に変換をしている。
                        # 損失和は「全データの損失/データ数」で計算されるため、
                        # 平均のままだと損失和を求めることができないため。
                        epoch_loss += loss.item() * inputs.size(0)

                        # 正解数の合計を更新
                        epoch_corrects += torch.sum(preds == labels.data)

                # epochごとのlossと正解率を表示
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    def inference(self, model, input_image: Image, is_transformed: bool = False, categories: dict = None):
        if not is_transformed:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        else:
            input_batch = input_image

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        if categories is None:
            # https://slundberg.github.io/shap/notebooks/ImageNet%20VGG16%20Model%20with%20Keras.html
            import requests
            r = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
            categories = r.json()
            top5_prob, top5_catid = torch.topk(probabilities, 5)
        else:
            k = len(categories.keys()) if len(categories.keys()) < 5 else 5
            top5_prob, top5_catid = torch.topk(probabilities, k)
        print('top5_prob: ', top5_prob)
        print('top5_catid: ', top5_catid)
        for i in range(top5_prob.size(0)):
            # print(categories[ str(top5_catid[i].item()) ])
            keys = [k for k, v in categories.items() if v == top5_catid[i].item()]
            print(top5_catid[i].item(), keys[0])

    # https://www.kaggle.com/code/pestipeti/simple-pytorch-inference/notebook

    def check_inference(self, class_to_idx: dict):
        BATCH_SIZE = 1
        device = 'cpu'

        test_dataset = MyDataset()
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=N_WORKERS,
            shuffle=False
        )

        self.model.eval()

        tk0 = tqdm(data_loader_test, desc="Iteration")

        results = []

        for step, batch in enumerate(tk0):
            inputs = batch["image"]
            image_ids = batch["image_id"]
            inputs = inputs.to(device, dtype=torch.float)
            self.inference(self.model, input_image=inputs, is_transformed=True, categories=class_to_idx)

    def save_model(self, model, model_dir: str, model_name: str = 'model.pth',
                   is_and_package_list: bool = True) -> None:
        """save train model
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        """
        model_path: str = os.path.join(model_dir, model_name)
        torch.save(self.get_train_model_weights(model=model), model_path)

        if is_and_package_list:
            pip_packages_path: str = os.path.join(model_dir, 'requirements.txt')
            cmd: str = f'pip freeze > {pip_packages_path}'
            os.system(cmd)

    def get_train_model_weights(self, model):
        return model.cpu().state_dict()

    def main(self, df: pd.DataFrame, model_name: str, num_epochs: int, is_face_recognition: bool = True):
        # if not torch.cuda.is_available():
        #     raise
        if df is None:
            self.df: pd.DataFrame = pd.read_csv('./tmp.csv')
        else:
            self.df: pd.DataFrame = df.copy()
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError

        self.model = get_model(model_name=model_name)
        logger.info('Start Setup')
        setup_data(df=self.df, is_face_recognition=is_face_recognition, timestamp=self.timestamp)

        _data_folder = f'./{self.timestamp}/face_recognition/' if is_face_recognition else f'./{self.timestamp}/original/'

        dataset = get_dataset(data_folder=_data_folder)
        train_dataset, valid_dataset = split_train_valid_dataset(dataset=dataset)
        self.class_to_idx = dataset.class_to_idx

        self.train(train_dataset=train_dataset, valid_dataset=valid_dataset, num_epochs=num_epochs)
        self.check_inference(class_to_idx=self.class_to_idx)
        self.save_model(model=self.model, model_dir=self.timestamp)


class FaceRecognition:
    def setup(self):
        cascade_path: str = 'haarcascade_frontalface_default.xml'
        r = requests.get(
            'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
            stream=True)

        with open(cascade_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

    def get_face_using_haarcascade_frontalface_default(self, df: pd.DataFrame, path_or_link: str = 'link'):
        """
        ref: https://zenn.dev/opamp/articles/73126cf8c0135d
        :return:
        """
        # 識別したい画像分だけ準備してください！
        cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

        data_list = [df[path_or_link].tolist()]

        # クラス名を記載してください！
        name_list = ["hoge"]
        for name, l in zip(name_list, data_list):
            count = 1
            for idx, path in enumerate(l):
                print(path)
                image = imread_web(path)
                if image is None:
                    print('Image is None')
                    continue
                gray = cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_BGR2GRAY)

                faces = cascade.detectMultiScale(gray)
                for (x, y, w, h) in faces:
                    cv2.imwrite(f"./cut/{name}_{count}.png", image[y:y + h, x:x + w])
                    count += 1
                if count >= 50:
                    break

    def main(self):
        df = pd.read_csv('./tmp.csv')
        self.setup()
        self.get_face_using_haarcascade_frontalface_default(df=df)
