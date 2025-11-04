import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset, MaskWiseVolumeDataset
import logging
import warnings

class KiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-54, 247)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class LiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2


class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-39, 204)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 68.45214
        self.global_std = 63.422806
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2


class ColonVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-57, 175)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class EndonasalVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        # MRI has much higher ranges than CT, because unlike CT they are not standardized
        self.intensity_range = (70.94605690002442, 2317.3609033203084)
        self.target_spacing = (0.5, 0.5, 0.5)
        self.global_mean = 432.88325227536376 # 123.61745421964683
        self.global_std = 350.76435733219495 # 246.33260633390296
        # Global Mean: 432.88325227536376, Global Std: 350.76435733219495
        self.spatial_index = [2, 1, 0]  # Unsure about this
        self.do_dummy_2D = False
        self.target_class = (1, 2)

class MaskWiseEndonasalVolumeDataset(MaskWiseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        # MRI has much higher ranges than CT, because unlike CT they are not standardized
        self.intensity_range = (70.94605690002442, 2317.3609033203084)
        self.target_spacing = (0.5, 0.5, 0.5)
        self.global_mean = 432.88325227536376 # 123.61745421964683
        self.global_std = 350.76435733219495 # 246.33260633390296
        # Global Mean: 432.88325227536376, Global Std: 350.76435733219495
        self.spatial_index = [2, 1, 0]  # Unsure about this
        self.do_dummy_2D = False
        self.target_class = (1, 2)

DATASET_DICT = {
    "kits": KiTSVolumeDataset,
    "lits": LiTSVolumeDataset,
    "pancreas": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
    "endonasal": EndonasalVolumeDataset,
    "maskwise_endonasal": MaskWiseEndonasalVolumeDataset,
}


def load_data_volume(
    *,
    data,
    path_prefix,
    batch_size,
    pkl_filename = "excluded_ids_split.pkl",
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:
        data_dir = os.path.join(path_prefix, pkl_filename)

    with open(data_dir, "rb") as f:
        d = pickle.load(f)[fold][split]

    img_files = [d[i][0] for i in d.keys()]
    print("Stored image files")
    # Process each mask string individually.
    seg_files = [
            seg_entry if isinstance(seg_entry, list) else [seg_entry]
            for i, seg_entry in ((i, d[i][1]) for i in d.keys())
                ]

    # This is the original code for loading the data
    # img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    # seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]

    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
        data_dir=path_prefix,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
        )
    return loader

def load_data_volume_maskwise(
    *,
    data,
    path_prefix,
    batch_size,
    pkl_filename,
    data_dir=None,
    split="train",
    target_class = None,
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    num_worker=4,
):
    if not path_prefix:
        raise ValueError("unspecified data directory")
    
    if data == "maskwise_endonasal":
        # The pkl file is inside the data_dir - use either "split_maskwise.pkl" or "excluded_ids_maskwise_split.pkl"
        # The excluded_ids_maskwise_split.pkl has MRI scans with bad quality excluded
        if not path_prefix:
            raise ValueError("unspecified data directory")
        if data_dir is None:
            data_dir = os.path.join(path_prefix, pkl_filename)
        
        with open(data_dir, "rb") as f:
            info = pickle.load(f)
        
        # Dictionaries
        image_paths = {}
        label_paths = {}
        mask_types = {}
        
        counter = 0
        print(info)
        if split == "train":
            for idx in info['tumor'][0][split]:
                image_paths[counter] = info['tumor'][0][split][idx][0]
                label_paths[counter] = info['tumor'][0][split][idx][1]
                mask_types[counter] = 0  # 0 for tumor
                counter += 1
            
            for idx in info['ica'][0][split]:
                image_paths[counter] = info['ica'][0][split][idx][0]
                label_paths[counter] = info['ica'][0][split][idx][1]
                mask_types[counter] = 1  # 1 for ICA
                counter += 1
        if split == "val" or split == "test":
            if target_class is None and split == "val":
                warnings.warn("target_class is None. Make sure that this is for training")
                for idx in info['tumor'][0][split]:
                    image_paths[counter] = info['tumor'][0][split][idx][0]
                    label_paths[counter] = info['tumor'][0][split][idx][1]
                    mask_types[counter] = 0  # 0 for tumor
                    counter += 1
            
                for idx in info['ica'][0][split]:
                    image_paths[counter] = info['ica'][0][split][idx][0]
                    label_paths[counter] = info['ica'][0][split][idx][1]
                    mask_types[counter] = 1  # 1 for ICA
                    counter += 1

            # For testing
            elif target_class == "tumor":
                for idx in info['tumor'][0][split]:
                    image_paths[counter] = info['tumor'][0][split][idx][0]
                    label_paths[counter] = info['tumor'][0][split][idx][1]
                    mask_types[counter] = 0  # 0 for tumor
                    counter += 1
            elif target_class == "ICA":
                for idx in info['ica'][0][split]:
                    image_paths[counter] = info['ica'][0][split][idx][0]
                    label_paths[counter] = info['ica'][0][split][idx][1]
                    mask_types[counter] = 1  # 1 for ICA
                    counter += 1
            else:
                raise ValueError(f"Invalid target class: {target_class} or split: {split}")
            

        dataset = DATASET_DICT[data](
            image_paths,
            label_paths,
            split=split,
            augmentation=augmentation,
            mask_type=mask_types,
            rand_crop_spatial_size=rand_crop_spatial_size,
            convert_to_sam=convert_to_sam,
            do_test_crop=do_test_crop,
            do_val_crop=do_val_crop,
            do_nnunet_intensity_aug=do_nnunet_intensity_aug,
            data_dir=path_prefix,
            )
            
        if deterministic:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
            )
        return loader
    else:
        raise ValueError("Invalid dataset, select maskwise_endonasal load_data_volume_maskwise(*args)")
    
    
