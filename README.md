# Individual Activity Anomaly Estimation in Operating Rooms Based on Time-Sequential Prediction
This research was commended with **The Second Prize of Best Student Papers on MEDINFO 2023** ! You can find further information at [here](https://medinfo2023.org/the-international-medical-informatics-association-imia-announces-medinfo-2023-best-paper-winners/).


We propose a semi-supervised individual activity anomaly estimation model based on time-sequential prediction using Generative Adversarial Network.
In this research, we compare two specific features that can be used as inputs to our method to acquire anomaly scores.

# Environments
- OS: Ubunts-20.04
- python: 3.8.13
- CUDA: 11.5

# Installation
```
pip install -U pip
pip install wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu115
```


# Training
```
python tools/train_individual.py [-h] -dd DATA_DIR -sl SEQ_LEN [-g [GPUS ...]] [-dt DATA_TYPE] [-msk]
```

optional arguments:
  -h, --help            show this help message and exit
  -dd DATA_DIR, --data_dir DATA_DIR
                        path of input data
  -sl SEQ_LEN, --seq_len SEQ_LEN
                        sequential length
  -g [GPUS ...], --gpus [GPUS ...]
                        gpu ids
  -dt DATA_TYPE, --data_type DATA_TYPE
                        'ganomaly': Input data type. Selected by 'global', 'local', or 'both', by defualt is 'local'.
  -msk, --masking       'ganomaly': Masking low confidence score keypoints


# Inference
```
python tools/inference_individual.py [-h] -dd DATA_DIR -sl SEQ_LEN [-g GPU] [-mv MODEL_VERSION] [-dt DATA_TYPE] [-msk]
```

optional arguments:
  -h, --help            show this help message and exit
  -dd DATA_DIR, --data_dir DATA_DIR
                        path of input data
  -sl SEQ_LEN, --seq_len SEQ_LEN
                        sequential length
  -g GPU, --gpu GPU     gpu id
  -mv MODEL_VERSION, --model_version MODEL_VERSION
                        model version
  -dt DATA_TYPE, --data_type DATA_TYPE
                        Input data type. Selected by 'global', 'local', 'local+bbox' or 'both', by defualt is 'local'.
  -msk, --masking       Masking low confidence score keypoints


# Reference
```
@proceedings{yokoyama_medinfo_2023,
  title={Individual Activity Anomaly Estimation in Operating Rooms Based on Time-Sequential Prediction},
  author={Yokoyama, Koji and Yamamoto, Goshiro and Liu, Chang and Kishimoto, Kazumasa and Mori, Yukiko and Kuroda, Tomohiro},
  journal={MEDINFO 2023—The Future Is Accessible},
  pages={284--288},
  year={2024},
  publisher={IOS Press}
}
```
