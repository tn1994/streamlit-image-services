import urllib

from PIL import Image


def get_input_image():
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    url, filename = (
        "https://asset.watch.impress.co.jp/img/wf/docs/1437/123/01_l.png", "01_l.png")

    try:
        urllib.URLopener().retrieve(url, filename)
    except BaseException:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    return input_image
