import json
import pathlib
import random
import subprocess as sp
import time
from concurrent import futures

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset import Datapool

CLASS_NAMES = ['bowling', 'ripping paper', 'playing xylophone', 'playing organ', 'playing bass guitar',
               'tapping guitar',
               'playing accordion', 'playing guitar', 'dribbling basketball', 'playing piano', 'playing bagpipes',
               'playing saxophone', 'playing harmonica', 'tickling', 'blowing nose', 'tapping pen', 'chopping wood',
               'blowing out candles', 'tap dancing', 'stomping grapes', 'playing clarinet', 'laughing',
               'playing trombone', 'shoveling snow', 'playing trumpet', 'playing violin', 'singing', 'shuffling cards',
               'playing keyboard', 'mowing lawn', 'playing drums']

class KineticsSound(Datapool):
    # train data : 14739
    # test data : 2594

    def __init__(self,
                 mode: str,
                 dataset_root_dir: str,
                 ):
        assert mode in ["train", "test"]
        self.mode = mode
        self.dataset_root_dir = pathlib.Path(dataset_root_dir) / mode
        assert self.dataset_root_dir.exists()
        self.data = {}
        audio_dir = self.dataset_root_dir / "audio"
        video_dir = self.dataset_root_dir / "video"

        if (self.dataset_root_dir.parent / f"{mode}.json").exists():
            with open(self.dataset_root_dir.parent / f"{mode}.json", 'r') as file:
                self.data = json.load(file)
        else:
            for audio_path in audio_dir.rglob("*.wav"):
                vid = audio_path.stem
                label = audio_path.parent.name
                video_path = video_dir / label / vid
                assert video_path.is_dir(), f"Cannot find {video_path}"
                self.data[vid] = {
                    "label": label,
                    "audio_path": str(audio_path),
                    "video_path": str(video_path)
                }
            with open(self.dataset_root_dir.parent / f"{mode}.json", 'w') as file:
                json.dump(self.data, file)

        self.all_ids = list(self.data.keys())
        self.all_ids.sort()
        random.Random(0).shuffle(self.all_ids)

        super(KineticsSound, self).__init__(self.all_ids, self.mode)

        self.train_transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.RandomCrop((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)),
        ])

    def extract_stft(self, audio_file_path):
        audio_file_path = pathlib.Path(audio_file_path)
        assert audio_file_path.exists()
        # audio
        # mean of spectrogram = -3.0352, std = 3.0475
        mean, std = -3.0352, 3.0475
        max, min = 4.74, -16.07
        sample, rate = librosa.load(audio_file_path, sr=16000, mono=True)

        if len(sample) >= 160000:
            sample = sample[:160000]
        else:
            sample = np.pad(sample, (0, 160000 - len(sample)), 'constant', constant_values=0)

        if self.mode == "train":
            start_point = random.randint(a=0, b=rate * 5)
            sample = sample[start_point:start_point + rate * 5]
        else:
            sample = sample[int(rate * 2.5): int(rate * 7.5)]
        sample[sample > 1.] = 1.
        sample[sample < -1.] = -1.

        spectrogram = librosa.stft(sample, n_fft=512, hop_length=159)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        spectrogram = (torch.from_numpy(spectrogram).unsqueeze(0) - mean) / std
        # spectrogram = (torch.from_numpy(spectrogram).repeat(3, 1, 1) - min) / (max - min)
        # spectrogram = torch.nn.functional.normalization(spectrogram)
        return spectrogram

    def load_frames(self, video_file_path):
        video_file_path = pathlib.Path(video_file_path)
        assert video_file_path.exists()
        # video
        frame_paths = [str(path) for path in video_file_path.glob("frame_*.jpg")]
        frame_paths.sort()
        if self.mode == "train":
            select_index = np.random.choice(10, size=5, replace=False)
            select_index.sort()
        else:
            select_index = range(1, 10, 2)
        frame_paths = [frame_paths[i] for i in select_index]
        frames = []

        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                frame = Image.open(f)
                frame = frame.convert("RGB")
            if self.mode == "train":
                frame = self.train_transform(frame)
            else:
                frame = self.val_transform(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # [channels, frames, H, W]
        return frames

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
            frames: size(channels=3, frames=10, H=224, W=224)
            sepctrograms: size(3, 257, 626)
            label: size()
        """
        sample_id = self.sample_ids[idx]
        spectrogram = self.extract_stft(self.data[sample_id]["audio_path"])
        frames = self.load_frames(self.data[sample_id]["video_path"])
        label = torch.tensor(CLASS_NAMES.index(self.data[sample_id]["label"]), dtype=torch.long)
        return frames, spectrogram, label

