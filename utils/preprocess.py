import os
import subprocess
from argparse import ArgumentParser
from glob import glob
from os.path import join, basename, exists
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import librosa
import soundfile as sf

def trim_silence(wav, top_db=60, frame_length=2048, hop_length=512):
    # -60dB = 0.001, -40dB = 0.01
    #return librosa.effects.trim(wav, top_db)
    intervals = librosa.effects.split(wav, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    buf = []
    for s, e in intervals:
        segment = wav[s:e]
        buf.append(segment)
    out = np.concatenate(buf, axis=0)
    return out, intervals

def get_loudest_part(y, frame_length=5 * 16000, hop_length=160):
    if len(y) < frame_length:
        return y
    # mse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)  # (1, t)
    x = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    mse = round(np.abs(x))
    idx = np.argmax(mse)
    return y[hop_length * idx:hop_length * idx + frame_length]

def process_one(f, tgt_dir):
    tgt_path = join(tgt_dir, basename(f))
    if exists(tgt_path):
        return
    wav, sr = librosa.load(f, sr=16000, mono=True)
    trimmed, _ = trim_silence(wav)
    # trimmed = get_loudest_part(wav)
    sf.write(tgt_path, trimmed, sr)

# trim silence
def process_silence(src_dir, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)
    files = glob(join(src_dir, '*.wav'))
    print(f'Trimming silence in audios from {src_dir} and writing to {tgt_dir}')
    parallel = Parallel(-1)
    parallel(delayed(process_one)(f, tgt_dir) for f in tqdm(files, total=len(files), dynamic_ncols=True, leave=True))

    print('Statistics of durations')
    before = []
    after = []
    for f in tqdm(files, total=len(files), dynamic_ncols=True, leave=True):
        tgt_path = join(tgt_dir, basename(f))
        before.append(librosa.get_duration(filename=f))
        after.append(librosa.get_duration(filename=tgt_path))
    before = np.array(before)
    after = np.array(after)
    print(f'Stats of durations before trmming: mean {before.mean()}s, std {before.std()}s, min {before.min()}s, max {before.max()}s')
    print(f'Stats of durations after trmming: mean {after.mean()}s, std {after.std()}s, min {after.min()}s, max {after.max()}s')
    # Stats of durations before trmming: mean 2.24s, std 1.03s, min 0.53s, max 9.96s
    # Stats of durations after trmming: mean 2.02s, std 1.01s, min 0.1s, max 9.84s

def process_norm(src_dir, tgt_dir):
    tgt_dir = '../data/wav_normed'
    os.makedirs(tgt_dir, exist_ok=True)
    print(f'Normlizing {src_dir} wavs and save to {tgt_dir}')
    files = glob(join(src_dir, '*.wav'))
    for f in tqdm(files):
        tgt_path = join(tgt_dir, basename(f))
        if exists(tgt_path):
            continue
        wav, sr = sf.read(f)
        if wav.ndim != 1:
            wav = wav[:, 0]
        normed = (wav - wav.mean()) / (wav.std() + 1e-6)

        sf.write(tgt_path, normed, sr)

def process_channel(src_dir, tgt_dir):
    tgt_dir = '../data/wav_singlech'
    os.makedirs(tgt_dir, exist_ok=True)
    print(f'Convert wav in {src_dir} into single channel and save in {tgt_dir}')
    files = glob(join(src_dir, '*.wav'))
    for f in tqdm(files):
        tgt_path = join(tgt_dir, basename(f))
        if exists(tgt_path):
            continue
        wav, sr = sf.read(f)
        if wav.ndim != 1:
            wav = wav[:, 0]
        sf.write(tgt_path, wav, sr)

def process_enhancement(src_dir, tgt_dir):
    tgt_dir = '../data/wav_enhanced'
    print('Denoising the wav, please wait...')
    cmd = f"python3 -m denoiser.enhance --dns64 --noisy_dir {src_dir} --out_dir {tgt_dir} --device cuda"
    subprocess.run(cmd, shell=True)
    # clean noisy wav
    subprocess.run(f"rm {join(tgt_dir, '*noisy.wav')}")
    # rename
    files = glob(join(tgt_dir, '*.wav'))
    assert len(files) == 59201, f"len(files): {len(files)} don't not equal to 59201"
    print('Rename...')
    for f in tqdm(files):
        fid = basename(f).split('_')[0]
        tgt_name = join(tgt_dir, f'{fid}.wav')
        os.rename(f, tgt_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--src_dir', type=str, default='../A-VB/audio/wav')
    parser.add_argument('-t', '--tgt_dir', type=str, default='../A-VB/audio/wav_trimmed')
    args, _ = parser.parse_known_args()
    process_silence(args.src_dir, args.tgt_dir)
