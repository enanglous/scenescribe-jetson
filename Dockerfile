FROM python:3.10 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    wget \
    qt5-qmake \
    qtbase5-dev \
    qttools5-dev-tools \
    qtdeclarative5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sip==6.7.7 pyqt5-sip==12.11.1 PyQt-builder==1.19.0

RUN wget https://files.pythonhosted.org/packages/source/P/PyQt5/PyQt5-5.15.9.tar.gz && \
    tar -xvzf PyQt5-5.15.9.tar.gz && \
    cd PyQt5-5.15.9 && \
    sip-install --confirm-license

FROM python:3.10

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

RUN apt-get update && apt-get install -y \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5network5 \
    libcap-dev \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/scenescribe/scenescribe

COPY requirements.txt .

RUN pip install -r ./requirements.txt

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

COPY ./models ./models

COPY ./src ./src
COPY ./utils ./utils
COPY ./recorded_audio.wav ./recorded_audio.wav

CMD ["python", "-m", "src.flask_backend.websockets_backend"]