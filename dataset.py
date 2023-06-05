from os.path import join, abspath
import hydra
from functools import reduce, partial
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import augment

from data_augment import ChainRunner
from utils.create_splits import LABEL_DICT
from utils.pl_utils import pl_print
from trainer.utils import to_tensor


class AVBDataset(Dataset):
    def __init__(self, phase, cfg):
        self.cfg = cfg
        self.phase = phase
        self.sr = cfg.sr
        self.wav = cfg.wav
        self.select = cfg.wav_select
        self.cv_fold = f"_{cfg.cv_fold}" if cfg.cv_fold >= 0 else ""
        self.classes = self.get_emotion_label_order()
        self.country_map = {"China": 0, "United States": 1, "South Africa": 2, "Venezuela": 3}
        self.high_map = {k: v for v, k in enumerate(LABEL_DICT["high"][1])}
        self.type_map = {"Cry": 0, "Gasp": 1, "Groan": 2, "Grunt": 3, "Laugh": 4, "Pant": 5, "Other": 6, "Scream": 7}

        ocwd = hydra.utils.get_original_cwd()
        self.data_dir = abspath(join(ocwd, self.cfg.data_dir))
        self.split_dir = abspath(join(ocwd, self.cfg.split_dir))

        df = pd.read_csv(join(self.data_dir, 'labels', LABEL_DICT[self.cfg.task][0] + '.csv'))
        country = pd.read_csv(join(self.split_dir, 'country.csv'))
        voc_type = pd.read_csv(join(self.data_dir, 'labels', 'type_info.csv'))
        two = pd.read_csv(join(self.data_dir, 'labels', 'two_info.csv'))
        # df = pd.merge(df, data, voc_type, two, on=['File_ID'])
        merged_df = [df, country, voc_type, two]
        if self.cfg.task == 'culture':
            merged_df += [pd.read_csv(join(self.data_dir, 'labels', 'high_info.csv'))]
        df = reduce(lambda x, y: pd.merge(x, y, on=['File_ID']), merged_df)

        selected_ids = pd.read_csv(join(self.split_dir, self.phase + f"{self.cv_fold}.csv"))
        # must reset_index to get item
        self.csv = df.loc[df['File_ID'].isin(selected_ids['File_ID'])].reset_index(drop=True)
        if cfg.debug:
            self.csv = self.csv.sample(2049).reset_index(drop=True)
        self.labels = self.csv.loc[:, self.classes].fillna(0)

        pl_print(f"{self.phase} data {len(self.csv)}")
        if self.phase == 'train' and self.cfg.augment.enable:
            chain = augment.EffectChain()
            chain.pitch(random.randint(-300, 300)).rate(self.sr)  # pitch shift
            chain.tempo(1.0 + 1.0 * (random.random() - 0.5) / 5)  # speed perturbation
            chain.dither()  # dither
            self.chain = ChainRunner(chain)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        if self.phase == 'test':
            out = {'fid': self.csv.loc[index, 'File_ID']}
        else:
            out = {
                'fid': self.csv.loc[index, 'File_ID'],
                'labels': self.csv.loc[index, self.classes].to_list(),
                'Country': self.country_map[self.csv.loc[index, 'Country']],
                'Voc_Type': self.type_map[self.csv.loc[index, 'Voc_Type']],
                'VA': self.csv.loc[index, ['Valence', 'Arousal']].to_list(),
                'emo': self.csv.loc[index, LABEL_DICT['high'][1]].to_list(),
            }
        if self.wav:
            wav = self.load_wav(out['fid'])
            out['wav'] = wav
        # print(out)
        # raise KeyboardInterrupt
        return out

    def collate_fn(self, batch):
        # batch: list of dict ==> dict of list
        batch_dic = {k: to_tensor([dic[k] for dic in batch]) for k in batch[0].keys()}
        for k, v in batch_dic.items():
            if isinstance(v, torch.DoubleTensor):
                batch_dic[k] = v.float()
        return batch_dic

    def get_emotion_label_order(self):
        mode = self.cfg.emotion_label_order
        # high performance to low
        h2l_order = ["Awe", "Surprise", "Amusement", "Fear", "Horror", "Sadness", "Distress", "Excitement", "Triumph", "Awkwardness"]
        # frquent to rare
        f2r_order = ["Amusement", "Surprise", "Fear", "Sadness", "Distress", "Excitement", "Awe", "Horror", "Awkwardness", "Triumph"]

        if mode == 'default' or self.cfg.task != 'high':
            return LABEL_DICT[self.cfg.task][1]  # default order used by the data
        elif mode == 'h2l':
            return h2l_order
        elif mode == 'l2h':
            return list(reversed(h2l_order))
        elif mode == 'f2r':
            return f2r_order
        elif mode == 'r2f':
            return list(reversed(f2r_order))
        else:
            raise ValueError(f"Unknown emotion label order mode: {mode}")

    def load_wav(self, fid):
        # TODO: truncate at the beginning, random or highest energy part?
        max_length = int(self.cfg.max_wav_length * self.sr)

        path = abspath(join(self.data_dir, 'audio', 'wav_trimmed', f"{fid[1:-1]}.wav"))
        if self.select == 'start':
            wav, sr = torchaudio.load(path, num_frames=max_length)
        else:
            wav, sr = torchaudio.load(path)
        # librosa loading time: 1.6561596393585205
        # torchaudio loading time: 0.009725332260131836

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if self.phase == 'test':
            return wav
        # wav = wav[0]  # mono channel, squeeze
        assert wav.ndim == 2 and wav.shape[0] == 1, f"Format error: {fid}, {wav.shape}"
        assert sr == self.sr, f"Sampling rate error: {fid}"
        if self.phase == 'train' and self.cfg.augment.enable:
            wav = self.augment(wav)

        # copy / trunct
        if self.cfg.wav_copy:
            while wav.shape[-1] < max_length:
                wav = torch.cat([wav, wav], -1)
        if self.select == 'start':
            wav = wav[:, :max_length]
        else:
            if wav.shape[-1] > max_length:
                idx = random.randint(0, wav.shape[-1] - max_length)
                wav = wav[:, idx:idx + max_length]
        assert wav.ndim == 2 and wav.shape[0] == 1 and wav.shape[-1] > 1600, f"Format error: {fid}, {wav.shape}"
        return wav

    def augment(self, wav):
        # pitch shift, speed perturb, spec aug for train, before clipping/padding
        wav = self.chain(wav)
        return wav


if __name__ == '__main__':
    dataset = AVBDataset('filelists/train.csv', '../data/wav', 16000)
    loader = DataLoader(dataset, 8, collate_fn=dataset.collate_fn)
    print(len(loader))
    for batch in loader:
        print(batch)
        break
