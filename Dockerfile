# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-gpu-images/python:v123

# ライブラリの追加インストール
RUN pip install -U pip && pip install  transformers==4.18.0 tokenizers==0.12.1 sentencepiece==0.1.96 datasets==1.18.3 iterative-stratification==0.1.7 torch-ema==0.3 typing-extensions==4.4.0 torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /tmp/working