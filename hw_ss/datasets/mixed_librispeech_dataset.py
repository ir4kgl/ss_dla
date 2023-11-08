import json
import os
from pathlib import Path
from glob import glob

import torchaudio

from hw_ss.base.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH


class MixedLibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir, index_dir, *args, **kwargs):
        self.indeces_map = {}
        self.next_label = 0
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

            target_id = int(target.split('/')[-1].split('_')[0])
            if not target_id in self.indeces_map.keys():
                self.indeces_map[target_id] = self.next_label
                self.next_label += 1
            index.append(
                {
                    "path_target": target,
                    "path_audio": mix,
                    "path_ref": ref,
                    "target_id": self.indeces_map[target_id]
                }
            )
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        path_audio = data_dict["path_audio"]
        path_ref = data_dict["path_ref"]
        path_target = data_dict["path_target"]
        target_id = data_dict["target_id"]
        audio_wave = self.load_audio(path_audio)
        ref_wave = self.load_audio(path_ref)
        target_wave = self.load_audio(path_target)
        audio_wave = self.process_wave(audio_wave)
        ref_wave = self.process_wave(ref_wave)
        target_wave = self.process_wave(target_wave)
        ref_len = ref_wave.shape[-1]
        audio_len = audio_wave.shape[-1]
        target_len = target_wave.shape[-1]
        return {
            "audio": audio_wave,
            "target": target_wave,
            "ref": ref_wave,
            "target_id": target_id,
            "target_len": target_len,
            "audio_len": audio_len,
            "ref_len": ref_len,
            "target_id": target_id
        }
