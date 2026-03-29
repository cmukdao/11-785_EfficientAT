import os
from functools import lru_cache
from torch.utils.data import Dataset as TorchDataset
import torch
import numpy as np
import pandas as pd
import torchaudio

from datasets.helpers.audiodatasets import PreprocessDataset, get_roll_func

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)


def _unique_existing_candidates(candidates):
    seen = set()
    unique = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _audio_dir_candidates(base_dir):
    if not base_dir:
        return []
    return [
        os.path.join(base_dir, "audio_32k"),
        os.path.join(base_dir, "audio"),
    ]


def _meta_csv_candidates(base_dir):
    if not base_dir:
        return []
    return [
        os.path.join(base_dir, "meta", "esc50.csv"),
        os.path.join(base_dir, "esc50.csv"),
    ]


def _resolve_audio_path():
    env_audio_dir = os.environ.get("EFFICIENTAT_ESC50_AUDIO_DIR")
    dataset_dir = os.environ.get("EFFICIENTAT_ESC50_DIR")
    candidates = _unique_existing_candidates(
        [env_audio_dir]
        + _audio_dir_candidates(dataset_dir)
        + _audio_dir_candidates(os.path.join(_REPO_ROOT, "datasets", "ESC-50"))
        + _audio_dir_candidates(os.path.join(_REPO_ROOT, "ESC-50"))
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find ESC-50 audio directory. "
        "Set EFFICIENTAT_ESC50_DIR to the dataset root, or set "
        "EFFICIENTAT_ESC50_AUDIO_DIR directly. Checked: "
        + ", ".join(candidates)
    )


def _resolve_meta_csv_path():
    env_meta_csv = os.environ.get("EFFICIENTAT_ESC50_META_CSV")
    dataset_dir = os.environ.get("EFFICIENTAT_ESC50_DIR")
    candidates = _unique_existing_candidates(
        [env_meta_csv]
        + _meta_csv_candidates(dataset_dir)
        + [
            os.path.join(_THIS_DIR, "esc50.csv"),
            os.path.join(_REPO_ROOT, "datasets", "ESC-50", "meta", "esc50.csv"),
            os.path.join(_REPO_ROOT, "ESC-50", "meta", "esc50.csv"),
        ]
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find ESC-50 metadata CSV. "
        "Set EFFICIENTAT_ESC50_META_CSV directly or place esc50.csv under "
        "EFFICIENTAT_ESC50_DIR/meta, EFFICIENTAT_ESC50_DIR/, or this repo's datasets/ directory. "
        "Checked: " + ", ".join(candidates)
    )


@lru_cache(maxsize=1)
def _get_dataset_config():
    return {
        'meta_csv': _resolve_meta_csv_path(),
        'audio_path': _resolve_audio_path(),
        'num_of_classes': 50
    }


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]


def pydub_augment(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


def load_audio(audio_path, sample_rate):
    waveform, original_sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=original_sample_rate, new_freq=sample_rate
        )
    return waveform.squeeze(0).float().numpy()


def _to_float32_tensor(value):
    if torch.is_tensor(value):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, f1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, f2, y2 = self.dataset[idx2]
            x1 = _to_float32_tensor(x1)
            x2 = _to_float32_tensor(x2)
            y1 = _to_float32_tensor(y1)
            y2 = _to_float32_tensor(y2)
            l = np.float32(np.random.beta(self.beta, self.beta))
            l = max(l, np.float32(1.0) - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = x1 * l + x2 * (np.float32(1.0) - l)
            x = x - x.mean()
            y = y1 * l + y2 * (np.float32(1.0) - l)
            return x, f1, y
        x, filename, y = self.dataset[index]
        return _to_float32_tensor(x), filename, _to_float32_tensor(y)

    def __len__(self):
        return len(self.dataset)


class AudioSetDataset(TorchDataset):
    def __init__(self, meta_csv, audiopath, fold, train=False, resample_rate=32000, classes_num=50,
                 clip_length=5, gain_augment=0):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.resample_rate = resample_rate
        self.meta_csv = meta_csv
        self.df = pd.read_csv(meta_csv)
        if train:  # training all except this
            print(f"Dataset training fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold != fold]
            print(f" for training remains {len(self.df)}")
        else:
            print(f"Dataset testing fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold == fold]
            print(f" for testing remains {len(self.df)}")

        self.clip_length = clip_length * resample_rate
        self.classes_num = classes_num
        self.gain_augment = gain_augment
        self.audiopath = audiopath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}
        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        row = self.df.iloc[index]

        waveform = load_audio(os.path.join(self.audiopath, row.filename), self.resample_rate)
        if self.gain_augment:
            waveform = pydub_augment(waveform, self.gain_augment)
        waveform = pad_or_truncate(waveform, self.clip_length).astype(np.float32, copy=False)
        target = np.zeros(self.classes_num, dtype=np.float32)
        target[int(row.target)] = 1.0
        return waveform.reshape(1, -1), row.filename, target


def get_base_training_set(resample_rate=32000, gain_augment=0, fold=1):
    dataset_config = _get_dataset_config()
    meta_csv = dataset_config['meta_csv']
    audiopath = dataset_config['audio_path']
    ds = AudioSetDataset(meta_csv, audiopath, fold, train=True,
                         resample_rate=resample_rate, gain_augment=gain_augment)
    return ds


def get_base_test_set(resample_rate=32000, fold=1):
    dataset_config = _get_dataset_config()
    meta_csv = dataset_config['meta_csv']
    audiopath = dataset_config['audio_path']
    ds = AudioSetDataset(meta_csv, audiopath, fold, train=False, resample_rate=resample_rate)
    return ds


def get_training_set(resample_rate=32000, roll=False, wavmix=False, gain_augment=0, fold=1):
    ds = get_base_training_set(resample_rate=resample_rate, gain_augment=gain_augment, fold=fold)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


def get_test_set(resample_rate=32000, fold=1):
    ds = get_base_test_set(resample_rate, fold=fold)
    return ds
