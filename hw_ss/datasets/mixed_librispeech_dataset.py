import json
import os
from pathlib import Path
from glob import glob

import torchaudio

from hw_ss.base.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH


class MixedLibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir, index_dir, *args, **kwargs):
        self._data_dir = Path(data_dir)
        self._index_dir = Path(index_dir)
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part

        ref_files = sorted(glob(os.path.join(split_dir, '*-ref.wav')))
        mix_files = sorted(glob(os.path.join(split_dir, '*-mixed.wav')))
        target_files = sorted(glob(os.path.join(split_dir, '*-target.wav')))

        for i in range(len(ref_files)):
            ref = ref_files[i]
            mix = mix_files[i]
            target = target_files[i]
            ref_info = torchaudio.info(ref)
            mix_info = torchaudio.info(mix)
            target_info = torchaudio.info(target)
            ref_length = ref_info.num_frames / ref_info.sample_rate
            audio_lenth = mix_info.num_frames / mix_info.sample_rate
            target_length = target_info.num_frames / target_info.sample_rate
            index.append(
                {
                    "path_target": target,
                    "path_audio": mix,
                    "path_ref": ref,
                    "target_length": target_length,
                    "audio_len": audio_lenth,
                    "ref_length": ref_length,
                }
            )
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        path_audio = data_dict["path_audio"]
        path_ref = data_dict["path_ref"]
        path_target = data_dict["path_target"]
        audio_wave = self.load_audio(path_audio)
        ref_wave = self.load_audio(path_ref)
        target_wave = self.load_audio(path_target)
        # audio_wave = self.process_wave(path_audio)
        # audio_wave = self.process_wave(path_audio)
        # audio_wave = self.process_wave(path_audio)
        return {
            "audio": audio_wave,
            "path_audio": path_audio,
            "target": target_wave,
            "path_target": path_target,
            "ref": ref_wave,
            "path_ref": path_ref,
        }
