FROM python:3.10.7

COPY requirements.txt .

# for face_recognition
RUN apt-get update && \
    apt install -y cmake && \
    apt-get install -y libgl1-mesa-dev

RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r requirements.txt
