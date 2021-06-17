import os
import json
import pathlib
from copy import deepcopy
from typing import Any, Dict, Union
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class RotationSegmentationDataset(Dataset):
    def __init__(self,
                 region: str,
                 root_path: Union[str, pathlib.Path],
                 split_and_mirror: bool = True,):
        self.region = region
        self.split_and_mirror = split_and_mirror
        self.root_path = root_path
        self.ndim = 3
        self.split_and_mirror = split_and_mirror
        self.data = self.parse_data_paths(self.root_path)

    def parse_data_paths(self, root_path: str) -> Dict[int, str]:
        if self.region.lower() == "hip":
            region = ["huefte", "huefte"]
        elif self.region.lower() == "knee":
            region = ["knie", "knie"]
        elif self.region.lower() == "ankle":
            region = ["osg", "knoechel"]
        else:
            raise ValueError

        data = []

        for pat in sorted(os.listdir(root_path)):
            for ser in sorted(
                    os.listdir(os.path.join(root_path, pat, "Study_0"))):
                with open(
                        os.path.join(root_path, pat, "Study_0", ser,
                                     "body_parts.json")) as file:
                    body_parts = json.load(file)
                for inst in sorted(
                        os.listdir(os.path.join(root_path, pat, "Study_0",
                                                ser)))[0:-1]:
                    if body_parts[inst] == region[0]:
                        sample = {
                            "data":
                            os.path.join(root_path, pat, "Study_0", ser, inst,
                                         "image.nii.gz"),
                            "label":
                            os.path.join(root_path, pat, "Study_0", ser, inst,
                                         region[1] + "_seg.nii.gz"),
                        }

                        if self.split_and_mirror:
                            data.append({
                                **sample, "side": "left",
                                'patient': int(pat.rsplit('_')[1])
                            })
                            data.append({
                                **sample, "side": "right",
                                'patient': int(pat.rsplit('_')[1])
                            })
                        else:
                            data.append({**sample, 'patient': int(pat.rsplit('_')[1])})

        return {idx: sample for idx, sample in enumerate(data)}

    def _read_sample_from_disk(self, sample: Any) -> dict:

        img = nib.load(sample["data"])
        seg = nib.load(sample["label"])

        spacing = img.header.get_zooms()

        if self.split_and_mirror:
            width = img.header.get_data_shape()[0] // 2
            if sample["side"] == "left":
                img = img.dataobj[width:]
                seg = seg.dataobj[width:]
                img = img[::-1]
                seg = seg[::-1]

            elif sample["side"] == "right":
                img = img.dataobj[:width]
                seg = seg.dataobj[:width]

            else:
                raise ValueError
        else:
            img = img.dataobj[:]
            seg = seg.dataobj[:]

        img = np.swapaxes(img, 0, 2).copy().astype(np.float32)
        seg = np.swapaxes(seg, 0, 2).copy().astype(np.float32)

        spacing = deepcopy(list(reversed(spacing)))
        img = torch.from_numpy(img).squeeze()[None]
        seg = torch.from_numpy(seg).squeeze()[None]

        return {
            "data": img,
            "label": seg,
            "orig_spacing": torch.tensor(spacing),
            "orig_size": torch.tensor(img.shape[1:]),
            **{k: v
               for k, v in sample.items() if k not in ('data', 'label')}
        }

    def __getitem__(self, index):
        return self._read_sample_from_disk(self.data[index])

    def __len__(self):
        return len(self.data)