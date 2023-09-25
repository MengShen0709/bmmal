import json
import cv2
import numpy as np
import pathlib
from tqdm import tqdm
from concurrent import futures
import subprocess as sp
import time
import librosa
import random
from dataset import Datapool
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import Counter


CLASS_NAMES = ['ice cream truck ice cream van', 'people babbling', 'mouse clicking', 'cat caterwauling',
               'cap gun shooting', 'eagle screaming', 'planing timber', 'goose honking', 'snake hissing', 'owl hooting',
               'cell phone buzzing', 'horse clip-clop', 'sheep bleating', 'swimming', 'donkey ass braying',
               'typing on computer keyboard', 'wind chime', 'child singing', 'people gargling',
               'cattle bovinae cowbell', 'mouse squeaking', 'underwater bubbling', 'firing cannon', 'playing bassoon',
               'arc welding', 'playing sitar', 'hedge trimmer running', 'subway metro underground', 'lawn mowing',
               'slot machine', 'playing darts', 'playing tabla', 'playing shofar', 'people eating noodle',
               'air conditioning noise', 'sharpen knife', 'people eating', 'driving motorcycle', 'baby laughter',
               'lions roaring', 'raining', 'sea lion barking', 'dog growling', 'tapping guitar', 'sliding door',
               'sloshing water', 'people battle cry', 'gibbon howling', 'playing guiro', 'hair dryer drying',
               'rowboat canoe kayak rowing', 'ambulance siren', 'woodpecker pecking tree', 'smoke detector beeping',
               'playing bagpipes', 'wind rustling leaves', 'reversing beeps', 'people eating crisps',
               'playing vibraphone', 'people crowd', 'playing volleyball', 'playing cornet', 'children shouting',
               'opening or closing drawers', 'playing washboard', 'cutting hair with electric trimmers', 'wind noise',
               'typing on typewriter', 'skiing', 'tornado roaring', 'hail', 'parrot talking', 'canary calling',
               'chinchilla barking', 'heart sounds heartbeat', 'singing choir', 'shot football', 'striking pool',
               'opening or closing car electric windows', 'playing harpsichord', 'train wheels squealing',
               'people slapping', 'people farting', 'printer printing', 'male speech man speaking', 'driving buses',
               'dog barking', 'airplane', 'playing harp', 'playing double bass', 'vehicle horn car horn honking',
               'hammering nails', 'baby babbling', 'playing gong', 'roller coaster running', 'police radio chatter',
               'playing harmonica', 'snake rattling', 'cow lowing', 'playing bugle', 'horse neighing', 'orchestra',
               'car engine idling', 'people hiccup', 'playing cymbal', 'bird squawking', 'race car auto racing',
               'playing drum kit', 'helicopter', 'ocean burbling', 'playing castanets', 'ice cracking',
               'striking bowling', 'chimpanzee pant-hooting', 'fly housefly buzzing', 'dog whimpering',
               'volcano explosion', 'bee wasp etc buzzing', 'lathe spinning', 'people shuffling', 'thunder',
               'footsteps on snow', 'playing djembe', 'crow cawing', 'basketball bounce', 'driving snowmobile',
               'otter growling', 'yodelling', 'male singing', 'playing timpani', 'playing congas', 'people booing',
               'dog bow-wow', 'people sneezing', 'police car (siren)', 'playing banjo', 'bowling impact',
               'magpie calling', 'splashing water', 'bird chirping tweeting', 'playing snare drum', 'beat boxing',
               'playing hockey', 'skidding', 'waterfall burbling', 'tractor digging', 'people slurping',
               'people clapping', 'playing timbales', 'playing zither', 'playing french horn', 'rope skipping',
               'elephant trumpeting', 'motorboat speedboat acceleration', 'people belly laughing', 'playing badminton',
               'mynah bird singing', 'pigeon dove cooing', 'car passing by', 'people screaming', 'playing oboe',
               'playing electronic organ', 'missile launch', 'ferret dooking', 'bird wings flapping',
               'child speech kid speaking', 'people cheering', 'francolin calling', 'telephone bell ringing',
               'playing hammond organ', 'people marching', 'door slamming', 'playing table tennis', 'chicken crowing',
               'playing piano', 'playing electric guitar', 'chicken clucking', 'people whispering',
               'electric grinder grinding', 'zebra braying', 'people whistling', 'ripping paper', 'air horn',
               'baby crying', 'chipmunk chirping', 'dinosaurs bellowing', 'dog baying', 'lions growling',
               'whale calling', 'female speech woman speaking', 'playing flute', 'warbler chirping', 'spraying water',
               'running electric fan', 'playing trumpet', 'pheasant crowing', 'cricket chirping', 'playing erhu',
               'playing accordion', 'playing bongo', 'civil defense siren', 'car engine starting', 'penguins braying',
               'scuba diving', 'tap dancing', 'playing steelpan', 'lighting firecrackers', 'playing tennis',
               'people eating apple', 'sea waves', 'wood thrush calling', 'electric shaver electric razor shaving',
               'goat bleating', 'lip smacking', 'duck quacking', 'foghorn', 'cat hissing', 'playing acoustic guitar',
               'coyote howling', 'playing clarinet', 'train whistling', 'playing didgeridoo', 'popping popcorn',
               'railroad car train wagon', 'eating with cutlery', 'playing tambourine', 'disc scratching',
               'people sobbing', 'people finger snapping', 'playing steel guitar slide guitar', 'playing synthesizer',
               'sailing', 'bouncing on trampoline', 'vacuum cleaner cleaning floors', 'strike lighter', 'metronome',
               'fire crackling', 'playing trombone', 'pig oinking', 'cattle mooing', 'chopping food', 'people humming',
               'airplane flyby', 'playing theremin', 'people giggling', 'people burping', 'playing cello',
               'playing saxophone', 'barn swallow calling', 'mosquito buzzing', 'people nose blowing',
               'alarm clock ringing', 'singing bowl', 'playing bass guitar', 'cat meowing', 'bull bellowing',
               'playing lacrosse', 'playing tympani', 'baltimore oriole calling', 'engine accelerating revving vroom',
               'playing glockenspiel', 'playing violin fiddle', 'blowtorch igniting', 'cat growling', 'pumping water',
               'opening or closing car doors', 'stream burbling', 'cupboard opening or closing', 'fire truck siren',
               'fireworks banging', 'people coughing', 'forging swords', 'elk bugling', 'church bell ringing',
               'golf driving', 'rapping', 'playing bass drum', 'machine gun shooting', 'cheetah chirrup',
               'mouse pattering', 'playing ukulele', 'cat purring', 'firing muskets', 'using sewing machines',
               'squishing water', 'alligators crocodiles hissing', 'people sniggering', 'chopping wood',
               'skateboarding', 'playing marimba xylophone', 'turkey gobbling', 'toilet flushing', 'female singing',
               'plastic bottle crushing', 'cuckoo bird calling', 'fox barking', 'black capped chickadee calling',
               'dog howling', 'playing squash', 'chainsawing trees', 'train horning', 'frog croaking',
               'playing mandolin', 'car engine knocking', 'eletric blender running', 'people running',
               'writing on blackboard with chalk', 'bathroom ventilation fan running', 'playing tuning fork']


class VggSound(Datapool):

    def __init__(self,
                 mode: str,
                 dataset_root_dir: str,
                 ):
        assert mode in ["train", "test"]
        self.mode = mode
        self.dataset_root_dir = pathlib.Path(dataset_root_dir)
        assert self.dataset_root_dir.exists()
        self.data = {}
        audio_dir = self.dataset_root_dir / "audio" / mode
        video_dir = self.dataset_root_dir / "frames" / mode

        json_path = self.dataset_root_dir.parent / f"vggsound_{mode}.json"
        if json_path.exists():
            print(f"load json from {json_path}")
            with open(json_path, 'r') as file:
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
            with open(json_path, 'w') as file:
                json.dump(self.data, file)
            print(len(self.data.keys()))

        self.all_ids = list(self.data.keys())
        self.all_ids.sort()
        random.Random(0).shuffle(self.all_ids)

        super(VggSound, self).__init__(self.all_ids, self.mode)

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
        mean, std = -3.0093, 2.7911
        max, min = 4.7137, -15.9638
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
