#!/usr/bin/env bash
data_names=$@
if [ $# -eq 0 ];
then
  echo "Slect data name from 'volleyball', 'collective-activity', 'ucf101'."
  return 2
fi

# collective activity
if [ printf '%s\n' "$@" | grep -qx "collective-activity" ];
then
  echo "Download collective-activity"
  mkdir data/collective-activity/
  cd data/collective-activity/
  wget http://cvgl.stanford.edu/data/ActivityDataset.zip
  wget http://cvgl.stanford.edu/data/ActivityDataset2.tar.gz
  unzip ActivityDataset.zip
  tar -xzvf ActivityDataset2.tar.gz
  rm ActivityDataset.zip
  rm ActivityDataset2.tar.gz
fi

# ucf101
if [ printf '%s\n' "$@" | grep -qx "ucf101" ];
then
  echo "Download ucf101"
  mkdir data/ucf101
  cd data/ucf101
  mkdir videos
  wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
  wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
  wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-DetectionTask.zip
  unrar e UCF101.rar videos/
  unzip UCF101TrainTestSplits-RecognitionTask.zip
  unzip UCF101TrainTestSplits-DetectionTask.zip
  mv ucfTrainTestlist/ recognition/
  mv UCF101_Action_detection_splits/ detection/
fi
