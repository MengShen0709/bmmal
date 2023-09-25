import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

# KINETICS-400
"""     videos      csv         replace
test:   38676       39805       0
train:  240258      246534      1392
val:    19879       19906       0
"""

SPLITS = ['test', 'train', 'val']

def load_label(csv):
    table = np.loadtxt(csv, skiprows=1, dtype=str, delimiter=',')
    return {k: v.replace('"', '') for k, v in zip(table[:, 1], table[:, 0])}

def collect_dict(path, split, replace_videos):
    split_video_path = path / "k400" / split
    split_csv = load_label(path / f'{split}.csv')
    split_videos = list(split_video_path.rglob('*.avi'))
    split_videos = {str(p.stem)[:11]:p for p in split_videos}
    # replace paths for corrupted videos
    match_dict = {k: replace_videos[k] for k in split_videos.keys() & replace_videos.keys()}
    print("replacement match :", len(match_dict))
    split_videos.update(match_dict)
    # collect videos with labels from csv: dict with {video_path: class}
    split_final = {split_videos[k]:split_csv[k] for k in split_csv.keys() & split_videos.keys()}
    return split_final

def parse_args():
    """
    Usage: python arrange_by_classes.py <path to downloaded dataset>
    """
    argparser = argparse.ArgumentParser('Arrange kinetics400 dataset by classes')
    argparser.add_argument('path', type=str, help='Path to downloaded dataset')
    args = argparser.parse_args()
    return args

def main(path):
    path = Path(path)
    assert path.exists(), f'Provided path:{path} does not exist'

    # collect videos in replacement
    replace = list((path / 'k400/replace').rglob('*.avi'))
    replace_videos = {str(p.stem)[:11]:p for p in replace}
    print("replacement videos number:", len(replace), list(replace_videos.items())[0])

    video_parent = path / 'video'

    for split in SPLITS:
        print(f'Working on: {split}')
        # create output path
        split_video_path = video_parent / split
        split_video_path.mkdir(exist_ok=True, parents=True)
        split_final = collect_dict(path, split, replace_videos)
        print(f'Found {len(split_final)} videos in split: {split}')

        labels = set(split_final.values())
        vid_pth = list(split_final.items())[0][0]
        label = list(split_final.items())[0][1]
        dst_vid = split_video_path / label / vid_pth.name

        # create label directories
        for label in labels:
            label_pth = split_video_path / label
            label_pth.mkdir(exist_ok=True, parents=True)
        for vid_pth, label in tqdm(split_final.items(), desc=f'Progress {split}'):
            dst_vid = split_video_path / label / vid_pth.name
            shutil.move(src=vid_pth, dst=dst_vid)
        # symlink videos to respective labels
        # for vid_pth, label in tqdm(split_final.items(), desc=f'Progress {split}'):
        #     dst_vid = split_video_path / label / vid_pth.name
        #     if dst_vid.is_symlink():
        #         dst_vid.unlink()
        #     dst_vid.symlink_to(vid_pth.resolve(), target_is_directory=False)

if __name__ == '__main__':
    path = "/workspace/Datasets/ks400/"
    main(path)