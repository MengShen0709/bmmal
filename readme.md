# Balanced Multimodal Active Learning (ACMMM-2023)

This is official implementation
for ["Towards Balanced Active Learning for Multimodal Classification"](https://arxiv.org/abs/2306.08306).

# Code Tree

- src
    - config (configuration for each experiment)
    - dataset (datasets)
    - exp (pytorch lightning module)
    - model (MM-models)
    - run (pytorch lightning trainer)
    - strategy (active learning sampling strategies)
    - utils

# Supported Active Learning Strategy

- bmmal
- random
- bald
- entropy
- coreset
- kmeans
- badge
- deepfool
- gcn

# Setup Anaconda

```shell
conda create -n mmal -m python=3.9
conda activate mmal
# we only test with pytorch version 1.13.1 and cuda 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# install the rest dependencies using pip
pip install -r requirements.txt
```

# Dataset

- UPMC_Food101
    1. Download dataset following this
       paper [Recipe recognition with large multimodal food dataset](https://ieeexplore.ieee.org/abstract/document/7169757)
    2. Process food101 using **preprocess.py** to get test.json and train.json files
    3. The final dataset file structure will be like:

```
├── UPMC_Food101
    ├── train.json
    ├── test.json
    ├── images
        ├── train
            ├── label_name
                ├── label_name_id.jpg
                ├── ...
        ├── test
            ├── label_name
                ├── label_name_id.jpg
                ├── ...
    ├── texts_txt
        ├── label_name
            ├── label_name_id.txt
            ├── ...
```

- KineticsSound
    1. Download dataset following [Kinetics Datasets Downloader](https://github.com/cvdfoundation/kinetics-dataset)
    2. Run **kinetics_convert_avi.py** to convert mp4 files into avi files.
    3. Run **kinetics_arrange_by_class.py** to organize the files.
    4. Run **extract_wav_and_frames.py** to extract wav files and 10 frame images as jpg.
    5. The final dataset file structure will be like:

```
├── kinetics_sound
    ├── my_train.txt
    ├── my_test.txt
    ├── train
        ├── video
            ├── label_name
                ├── vid_start_end
                    ├── frame_0.jpg
                    ├── frame_1.jpg
                    ├── ...
                    ├── frame_9.jpg
        ├── audio
            ├── label_name
                ├── vid_start_end.wav
                ├── ...
    ├── test
        ├── ...
```

- VGGSound
    1. Download dataset following [VGGSound](https://github.com/hche11/VGGSound)
    2. Run **vggsound_convert_avi.py** to convert mp4 files into avi files.
    3. Run **extract_wav_and_frames.py** to extract wav files and 10 frame images as jpg.
    4. The final dataset file structure will be like:

```
├── vggsound
    ├── vggsound.csv
    ├── video
        ├── train
            ├── label_name
                ├── vid_start_end.avi
        ├── test
            ├── ...
    ├── frames
        ├── train
            ├── label_name
                ├── vid_start_end
                    ├── frame_0.jpg
                    ├── frame_1.jpg
                    ├── ...
                    ├── frame_9.jpg
        ├── test
            ├── ...
    ├── audio
        ├── train
            ├── label_name
                ├── vid_start_end.wav
        ├── test
            ├── ...
```

# Run the Experiments

```shell
cd mmal
export PYTHONPATH=$PWD
python run/runner.py -s {strategy} --seed {random_seed} -c {config_file} -d {cuda_device_index} -r {al_iteration} 
```

Currently, we only support to run experiments on **Single GPU** card.

## (Important) For fair comparison 

To make sure each strategy begins with the same initialization, we highly recommend to start with the copy of the model
trained with random sampling.

For example, if you want to examine performance of bmmal and badge:

```shell
# run the first iteration of active learning using random sampling
python run/runnner.py -s random --seed 1000 -c config/food101.yml -d 0 -r 0
# keep a copy of first iteration
cp -r logs/food101/food101-random-1000/version_0 logs/food101/food101-random-initialized-1000/version_0 
cp -r logs/food101/food101-random-1000/task_model.ckpt logs/food101/food101-random-initialized-1000/task_model.ckpt

# get a copy and renamed it as bmmal
cp -r logs/food101/food101-random-initialized-1000/version_0 logs/food101/food101-bmmal-1000/version_0 
cp -r logs/food101/food101-random-initialized-1000/task_model.ckpt logs/food101/food101-bmmal-1000/task_model.ckpt
# start bmmal sampling and training for second iteration
python run/runnner.py -s bmmal --seed 1000 -c config/food101.yml -d 0 -r 1

# get a copy and renamed it as badge
cp -r logs/food101/food101-random-initialized-1000/version_0 logs/food101/food101-badge-1000/version_0 
cp -r logs/food101/food101-random-initialized-1000/task_model.ckpt logs/food101/food101-badge-1000/task_model.ckpt
# start badge sampling and training for second iteration
python run/runnner.py -s badge --seed 1000 -c config/food101.yml -d 0 -r 1
```

By doing so, we can fairly compare the performance among different strategies with the same initialized iteration zero.

# Monitor Active Learning in Tensorboard

## Log File Tree

If we run active learning loop for 5 rounds, we will see version_0 to version_4 storing logging files for each
round.

- logs
    - {logger_save_dir}
        - {dataset_name}-{strategy}-{random_seed}
            - version_{al_iteration}
                - metrics.csv
                - al_metrics.csv # it stores metrics values in csv format
                - tf
                    - tf.events

```shell
tensorboard --logdir logs/{logger_save_dir}
```

# Cite

If you find this code helpful, please cite our paper:

```
@article{shen2023towards,
  title={Towards Balanced Active Learning for Multimodal Classification},
  author={Shen, Meng and Huang, Yizheng and Yin, Jianxiong and Zou, Heqing and Rajan, Deepu and See, Simon},
  journal={arXiv preprint arXiv:2306.08306},
  year={2023}
}
```
