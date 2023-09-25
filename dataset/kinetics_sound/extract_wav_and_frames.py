import pathlib
import subprocess as sp
import time
from concurrent import futures
import cv2
import numpy as np
from tqdm import tqdm


def load_csv(csv):
    # load video_ids
    table = np.loadtxt(csv, dtype=str, delimiter=',')
    return [k[:11] for k in table[:, 0]]


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


def extract_wav_from_avi():
    for split in splits:
        kinetics_sound_audio_dir = kinetics_sound_dir / split / "audio"
        kinetics_sound_audio_dir.mkdir(parents=True, exist_ok=True)

        kinetics_400_video_paths = (kinetics_400_dir / "video" / split).rglob("*.avi")
        kinetics_400_dict = {}
        for path in tqdm(kinetics_400_video_paths):
            label = path.parent.name
            vid = path.name[:11]
            stem = path.stem
            kinetics_400_dict[vid] = {"label": label,
                                      "video_path": path,
                                      "stem": stem}

        kinetics_sound_vids = load_csv(str(kinetics_400_dir / "my_{split}.txt"))

        # Match to raw kinetics 400 data
        no_such_video = 0
        kinetics_400_train_vids = list(kinetics_400_dict.keys())
        kinetics_sound_dict = {}
        for vid in tqdm(kinetics_sound_vids):
            if vid in kinetics_400_train_vids:
                label = kinetics_400_dict[vid]["label"]
                video_path = kinetics_400_dict[vid]["video_path"]
                audio_path = kinetics_sound_audio_dir / kinetics_400_dict[vid]["label"]
                audio_path.mkdir(exist_ok=True)
                audio_path = audio_path / f"{kinetics_400_dict[vid]['stem']}.wav"
                kinetics_sound_dict[vid] = {"label": label,
                                            "video_path": video_path,
                                            "audio_path": audio_path}
            else:
                no_such_video += 1

        # Start converting audio into wav files
        total = len(kinetics_sound_dict)
        complete = 0
        start_time = time.time()
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_audio = {executor.submit(extract_audio, *[v["video_path"], v["audio_path"]]):
                                   v for v in kinetics_sound_dict.values()}
            for future in futures.as_completed(future_to_audio):
                end_time = time.time()
                complete += 1
                try:
                    print(f">>> progress {complete / total * 100:.2f}%, "
                          f"ETA {(total - complete) * ((end_time - start_time) / complete) / 60:.2f} M, "
                          f"Used {(end_time - start_time) / 60:.2f} M")
                except Exception as exc:
                    print(exc)
                else:
                    pass

        print(f"Success: {complete}, Total: {total}")


def extract_frames(video_file_path: pathlib.Path, frames_save_path: pathlib.Path):
    video = cv2.VideoCapture(str(video_file_path))

    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames == 0:
        total_frames = 100
    fps = total_frames // 10
    video_duration = 10

    more_frames, frame = video.read()
    frames = [frame]
    count = 0
    while more_frames:
        count += 1
        if count % fps == 0 and count < fps * video_duration:
            frames.append(frame)
        more_frames, frame = video.read()

    if len(frames) != 10:
        print(f"Video path: {video_file_path}, Sample frames: {len(frames)}, "
              f"Total frames: {total_frames}, fps: {fps}, video duration: {video_duration} s")
    else:
        frames_save_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imencode('.jpg', frame)[1].tofile(str(frames_save_path / f"frame_{i}.jpg"))


def extract_frames_from_avi():
    for split in splits:
        kinetics_sound_vids = load_csv(str(kinetics_400_dir / "my_{split}.txt"))

        kinetics_sound_video_test_dir = kinetics_400_dir / "video" / split
        extract_frames_dst_dir_path = kinetics_sound_dir / split / "video"
        video_src_paths = []
        for video_src_path in tqdm(kinetics_sound_video_test_dir.rglob("*.avi")):
            if video_src_path.name[:11] in kinetics_sound_vids:
                video_src_paths.append(video_src_path)

        video_dst_paths = [extract_frames_dst_dir_path / src.parent.name / src.stem for src in video_src_paths]

        # Start converting audio into wav files
        total = len(video_src_paths)
        complete = 0
        start_time = time.time()
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_frames = {executor.submit(extract_frames, *[src, dst]):
                                    [src, dst] for src, dst in zip(video_src_paths, video_dst_paths)}
            for future in futures.as_completed(future_to_frames):
                end_time = time.time()
                complete += 1
                try:
                    print(f">>> progress {complete / total * 100:.2f}%, "
                          f"ETA {(total - complete) * ((end_time - start_time) / complete) / 60:.2f} M, "
                          f"Used {(end_time - start_time) / 60:.2f} M", end='\r', flush=True)
                except Exception as exc:
                    print(exc)
                else:
                    pass


if __name__ == "__main__":
    splits = ["train", "test"]
    kinetics_400_dir = pathlib.Path("/workspace/Datasets/ks400")
    # prepare my_{split}.txt under kinetics_sound_dir before processing
    kinetics_sound_dir = pathlib.Path("/workspace/Datasets/kinetics_sound")
    extract_frames_from_avi()
    extract_wav_from_avi()
