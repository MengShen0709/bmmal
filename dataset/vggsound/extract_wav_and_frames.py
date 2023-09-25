import pathlib
import subprocess as sp
from concurrent import futures
import cv2
from tqdm import tqdm


def extract_audio(video_path, audio_path):
    audio_convert_args = ' '.join(['ffmpeg', '-y', '-loglevel', 'error',
                                   '-i', f'\"{str(video_path)}\"',  # Specify the input audio URL
                                   '-f', 'wav',  # Specify the format (container) of the audio
                                   '-ar', '16000',  # Specify the sample rate
                                   '-ac', '1',  # mono channel
                                   '-acodec', 'pcm_s16le',  # Specify the output encoding
                                   f'\"{str(audio_path)}\"'])
    proc = sp.Popen(audio_convert_args, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, universal_newlines=True,
                    encoding='ascii')

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(stderr)


def extract_frames(video_file_path: pathlib.Path, frames_save_path: pathlib.Path):
    uniform_sample_frames = 10
    video = cv2.VideoCapture(str(video_file_path))
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_interval = int(total_frames / uniform_sample_frames)

    more_frames, frame = video.read()
    frames = [frame]
    count = 0
    while more_frames and len(frames) < uniform_sample_frames:
        count += 1
        if count % frames_interval == 0:
            frames.append(frame)
        more_frames, frame = video.read()

    if len(frames) != 10:
        print(f"Video path: {video_file_path}")
    else:
        frames_save_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imencode('.jpg', frame)[1].tofile(str(frames_save_path / f"frame_{i}.jpg"))


def extract():
    for split in splits:
        # src path : vggsound_root_dir / "video" / "split" / "label" / *.avi
        video_paths = list((vgg_sound_dir / "video" / split).rglob("*.avi"))
        # frame path : vggsound_root_dir / "video" / "split" / "label" / "vid" / *.jpg
        frame_paths = [vgg_sound_dir / "frames" / split / video_path.parent.name / video_path.stem for video_path in video_paths]
        # audio path : vggsound_root_dir / "video" / "split" / "label" / "vid" / *.jpg
        audio_paths = [vgg_sound_dir / "frames" / split / video_path.parent.name / video_path.stem for video_path in video_paths]

        # Start extracting 10 frame jpg files
        frame_tqdm = tqdm(total=len(video_paths))
        with futures.ProcessPoolExecutor(max_workers=8) as executor:
            future_to_frames = {executor.submit(extract_frames, *[video_path, frame_path]):
                                    [video_path, frame_path] for video_path, frame_path in zip(video_paths, frame_paths)}
            for future in futures.as_completed(future_to_frames):
                frame_tqdm.update(1)

        # Start extracting wav files
        audio_tqdm = tqdm(total=len(video_paths))
        with futures.ProcessPoolExecutor(max_workers=8) as executor:
            future_to_frames = {executor.submit(extract_audio, *[video_path, audio_path]):
                                    [video_path, audio_path] for video_path, audio_path in zip(video_paths, audio_paths)}
            for future in futures.as_completed(future_to_frames):
                audio_tqdm.update(1)


if __name__ == "__main__":
    splits = ["train", "test"]
    vgg_sound_dir = pathlib.Path("/workspace/Datasets/vggsound")
    assert vgg_sound_dir.exists()
    extract()
