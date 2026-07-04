"""
Each item is a sliding window of `seq_len` frames:
    frames        [T, 3, 224, 224]  (after AutoImageProcessor)
    binary_label  [T]               (0 normal / 1 error)
    multi_label   [T, 7]            (one-hot, class 0 = normal)
    frame_paths, video_name, frame_name
"""

import os
import csv
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from typing import List


class RARPFrameDataset(Dataset):
    def __init__(self,
                 image_roots: List[str],
                 label_roots: List[str],
                 split_file: str,
                 seq_len: int = 32,
                 stride: int = 16,
                 num_classes: int = 7,
                 transform=None,
                 image_size=(1080, 1920),
                 use_cv2=True):
        self.image_roots = image_roots
        self.label_roots = label_roots
        self.seq_len = seq_len
        self.stride = stride
        self.num_classes = num_classes
        self.transform = transform
        self.image_size = image_size
        self.use_cv2 = use_cv2

        self.binary_zero = torch.tensor(0.0, dtype=torch.float32)
        self.binary_one = torch.tensor(1.0, dtype=torch.float32)

        with open(split_file, "r") as f:
            self.video_list = [line.strip() for line in f if line.strip()]

        print(f"[RARPFrameDataset] Building samples from {len(self.video_list)} videos ({split_file})...")
        self.samples = self._build_samples()
        print(f"[RARPFrameDataset] Initialized with {len(self.samples)} samples")

        self._zero_img = np.zeros((*self.image_size, 3), dtype=np.uint8)

    @staticmethod
    def _find_existing_path(roots: List[str], name: str):
        for root in roots:
            candidate = os.path.join(root, name)
            if os.path.exists(candidate):
                return candidate
        return None

    def _build_samples(self):
        samples = []
        for video_name in self.video_list:
            video_dir = self._find_existing_path(self.image_roots, video_name)
            if video_dir:
                video_dir = video_dir + '/frame_10HZ'
            label_file = self._find_existing_path(self.label_roots, f"{video_name}.csv")
            if video_dir is None or label_file is None:
                continue

            label_map = {}
            with open(label_file, newline='') as f:
                for row in csv.DictReader(f):
                    frame = row['filename']
                    label_str = str(row['error']).replace(" ", "")
                    if label_str:
                        label_map[frame] = [int(x) for x in label_str.split(',')]

            all_frames = os.listdir(video_dir)
            frames = sorted([f for f in all_frames if f in label_map])
            if len(frames) < 1:
                continue

            for i in range(self.seq_len - 1, len(frames), self.stride):
                start_idx = i - self.seq_len + 1
                frame_seq, label_seq = [], []
                for j in range(start_idx, i + 1):
                    if j < 0:
                        frame_seq.append('None')
                        label_seq.append([])
                    else:
                        frame_seq.append(os.path.join(video_dir, frames[j]))
                        label_seq.append(label_map[frames[j]])

                if any(255 in lbl for lbl in label_seq) or any(254 in lbl for lbl in label_seq):
                    continue
                samples.append((frame_seq, label_seq, video_name, frames[i]))

        return samples

    def _load_image(self, path):
        if path == 'None':
            return self._zero_img
        if self.use_cv2:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: failed to load {path}, using zero image")
                img = self._zero_img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            try:
                img = np.array(Image.open(path))
            except Exception as e:
                print(f"Error loading {path}: {e}, using zero image")
                img = self._zero_img
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label_lists, video_name, frame_name = self.samples[idx]

        frames = [self._load_image(p) for p in frame_paths]

        multi_labels, binary_labels = [], []
        for label_list in label_lists:
            ml = torch.zeros(self.num_classes, dtype=torch.float32)
            for l in label_list:
                if 0 <= l < self.num_classes:
                    ml[l] = 1.0
            multi_labels.append(ml)

            if label_list:
                # class 0 = normal; ml.sum() == 1 means only normal is set
                binary_labels.append(
                    self.binary_zero if (0 in label_list and ml.sum() == 1)
                    else self.binary_one
                )
            else:
                binary_labels.append(self.binary_zero)

        frames = np.stack(frames)
        if self.transform:
            frames = self.transform(images=frames, return_tensors="pt")["pixel_values"]  # [T,3,224,224]

        multi_label_tensor = torch.stack(multi_labels)   # [T, num_classes]
        binary_label_tensor = torch.stack(binary_labels)  # [T]

        return frames, binary_label_tensor, multi_label_tensor, frame_paths, video_name, frame_name
