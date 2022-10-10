import re


def get_only_number(text: str) -> str:
    number = re.search(r'[\d]+', text)
    return number.group()


def get_only_version(text: str) -> str:
    version = re.search(r'[\d.]+[\w.]+', text)
    return version.group()
