import os
import pathlib

from tqdm import tqdm
from concurrent import futures
import subprocess as sp
import time


def convert(mp4_file_path):
    avi_file_path = mp4_file_path.replace(".mp4", ".avi")
    args = ' '.join(['ffmpeg', '-y', '-loglevel', 'error',
                     '-i', f'{mp4_file_path}',  # Specify the input
                     '-c:v', 'mpeg4',
                     '-filter:v', '\"scale=min(iw\,(256*iw)/min(iw\,ih)):-1\"',
                     '-b:v', '512k',
                     f'{avi_file_path}'])
    proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, universal_newlines=True, encoding='ascii')
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise ValueError(f"Video corrupted")

    return "Success"


if __name__ == "__main__":
    root_path = pathlib.Path("/workspace/Datasets/vggsound/video/")
    mp4_file_paths = list(root_path.rglob(".mp4"))
    with futures.ProcessPoolExecutor(max_workers=4) as executor:
        # Start the load operations and mark each future with its URL
        future_to_video = {executor.submit(convert, mp4_file_path): mp4_file_path for mp4_file_path in mp4_file_paths}
        for future in tqdm(futures.as_completed(future_to_video), total=len(mp4_file_paths)):
            pass
