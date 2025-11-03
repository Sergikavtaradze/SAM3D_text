import pickle
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    NormalizeIntensityd,
    MapTransform,
    RandSpatialCropd,
    CenterSpatialCropd,
)
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import os
# import matplotlib.pyplot as plt


class binarizeLabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            threshold: float = 0.5,
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d


class BaseVolumeDataset(Dataset):
    def __init__(
            self,
            image_paths,
            label_meta,
            augmentation,
            dataset_name = "endonasal",
            split="train",
            rand_crop_spatial_size=(96, 96, 96),
            convert_to_sam=True,
            do_test_crop=True,
            do_val_crop=True,
            do_nnunet_intensity_aug=True,
            data_dir=None,
    ):
        super().__init__()
        self.img_dict = image_paths
        self.label_dict = label_meta
        self.aug = augmentation
        self.split = split
        self.dataset_name = dataset_name
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug
        self.do_val_crop = do_val_crop
        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None

        self._set_dataset_stat()
        self.transforms = self.get_transforms()
        self.data_dir = data_dir

    def _set_dataset_stat(self):
        pass

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        # Construct the full image path using the base data directory.
        img_rel_path = self.img_dict[idx]
        img_path = os.path.join(self.data_dir, img_rel_path)

        # Similarly, construct full paths for both mask files.
        mask_paths_rel = self.label_dict[idx]  # This should be a list: [tumor, ica]
        
        # Some are strings which means that mask_paths iterates over a string --> self.data_dir/m is the mask path
        if isinstance(mask_paths_rel, str): 
            mask_paths_rel = [mask_paths_rel]

        mask_paths = [os.path.join(self.data_dir, mp) for mp in mask_paths_rel]
        
        # Load image volume
        img_vol = nib.load(img_path)
        img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        img_spacing = tuple(np.array(img_vol.header.get_zooms())[self.spatial_index])

        # Replace NaNs with 0
        img[np.isnan(img)] = 0


        # Load both mask volumes
        if self.dataset_name == "endonasal" or self.dataset_name in ["kits", "pancreas", "lits", "colon"]:
            mask_vol = nib.load(mask_paths[0])
            seg = mask_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
            seg[np.isnan(seg)] = 0
            seg = seg[None, ...]
        elif self.dataset_name == "maskwise_endonasal":
            tumor_vol = nib.load(mask_paths[0])
            ica_vol   = nib.load(mask_paths[1])
            tumor = tumor_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
            ica   = ica_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
            tumor[np.isnan(tumor)] = 0
            ica[np.isnan(ica)] = 0
            # Stack the masks to create a multi-channel segmentation (shape: [2, ...])
            seg = np.stack([tumor, ica], axis=0).astype(np.float32)

        # seg = np.stack([tumor, ica], axis=0).astype(np.float32)

        # --- Resampling / Interpolation ---
        if (np.max(img_spacing) / np.min(img_spacing) > 8) or (
                np.max(self.target_spacing / np.min(self.target_spacing)) > 8):
            # Process the image slice-wise (assumes img shape: [D, H, W])
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),  # shape: (D, 1, H, W)
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bilinear",
                align_corners=False,
            )

            seg_channels = []
            for c in range(seg.shape[0]):
                seg_channel_tensor = F.interpolate(
                    input=torch.tensor(seg[c, :, :, :])[:, None, :, :],  # shape: (D, 1, H, W)
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                    mode="bilinear",
                    align_corners=False,
                )
                seg_channels.append(seg_channel_tensor)
            seg_tensor = torch.cat(seg_channels, dim=1)

            # Final interpolation along the slice dimension
            img = (
                F.interpolate(
                    input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .numpy()
            )
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .numpy()
                )
        else:
            img = (
                F.interpolate(
                    input=torch.tensor(img[None, None, :, :, :]),
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .numpy()
            )
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=torch.tensor(seg[None, :, :, :, :]),
                        scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                        mode="trilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .numpy()
                )
        # print(f"This is the image shape: {img.shape}")
        # print(f"This is the seg shape: {seg.shape}")
        # logger.info(f"This is the image shape: {img.shape}")
        # logger.info(f"This is the seg shape: {seg.shape}")
        # The transforms might return a list of dictionaries or a single dictionary
        transform_result = self.transforms({"image": img, "label": seg})
        # Handle the case where transform_result is a list
        if isinstance(transform_result, list):
            trans_dict = transform_result[0]  # Get the first item from the list
        else:
            trans_dict = transform_result
        
        img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        seg_aug = seg_aug.squeeze()

        img_aug = img_aug.repeat(3, 1, 1, 1)

        return img_aug, seg_aug, np.array(img_vol.header.get_zooms())[self.spatial_index]

    def get_transforms(self):
        transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[0],
                b_max=self.intensity_range[1],
                clip=True,
            ),
        ]

        if self.split == "train":
            transforms.extend(
                [
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=20,
                        prob=0.5,
                    ),
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0],
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                ]
            )

            if self.do_dummy_2D:
                transforms.extend(
                    [
                       RandRotated(
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            keep_size=False,
                                ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=[1, 0.9, 0.9],
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )
            else:
                transforms.extend(
                    [
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=0.85,
                            max_zoom=1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    binarizeLabeld(keys=["label"]),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),
                    
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                ]
            )
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    binarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.split == "val"):
            transforms.extend(
                [
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),

                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    binarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    binarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms

class MaskWiseVolumeDataset(Dataset):
    def __init__(
            self,
            image_paths,
            label_paths,
            augmentation,
            split="train",
            mask_type=None,
            rand_crop_spatial_size=(96, 96, 96),
            convert_to_sam=True,
            do_test_crop=True,
            do_val_crop=True,
            do_nnunet_intensity_aug=True,
            data_dir=None,
    ):
        super().__init__()
        self.img_dict = image_paths
        self.label_dict = label_paths
        self.aug = augmentation
        self.split = split
        self.mask_type = mask_type  # Store mask type
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug
        self.do_val_crop = do_val_crop
        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None

        self._set_dataset_stat()
        self.transforms = self.get_transforms()
        self.data_dir = data_dir

    def _set_dataset_stat(self):
        pass

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        img_rel_path = self.img_dict[idx]
        img_path = os.path.join(self.data_dir, img_rel_path)
        mask_rel_path = self.label_dict[idx]
        
        # Handle both string and list formats
        if isinstance(mask_rel_path, list):
            mask_rel_path = mask_rel_path[0]  # Use the first path if it's a list
        
        mask_path = os.path.join(self.data_dir, mask_rel_path)
        
        # Load image volume
        img_vol = nib.load(img_path)
        img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        img_spacing = tuple(np.array(img_vol.header.get_zooms())[self.spatial_index])

        # Load mask volume
        mask_vol = nib.load(mask_path)
        mask = mask_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)

        # Replace NaNs with 0
        img[np.isnan(img)] = 0
        mask[np.isnan(mask)] = 0

        background = 1.0 - mask
        
        # Stack [background, foreground]
        seg = np.stack([background, mask], axis=0).astype(np.float32)

        # --- Resampling / Interpolation ---
        if (np.max(img_spacing) / np.min(img_spacing) > 8) or (
                np.max(self.target_spacing / np.min(self.target_spacing)) > 8):
            # Process the image slice-wise (assumes img shape: [D, H, W])
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),  # shape: (D, 1, H, W)
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bilinear",
                align_corners=False,
            )

            seg_channels = []
            for c in range(seg.shape[0]):
                seg_channel_tensor = F.interpolate(
                    input=torch.tensor(seg[c, :, :, :])[:, None, :, :],  # shape: (D, 1, H, W)
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                    mode="bilinear",
                    align_corners=False,
                )
                seg_channels.append(seg_channel_tensor)
            seg_tensor = torch.cat(seg_channels, dim=1)

            img = (
                F.interpolate(
                    input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .numpy()
            )
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .numpy()
                )
        else:
            img = (
                F.interpolate(
                    input=torch.tensor(img[None, None, :, :, :]),
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .numpy()
            )
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=torch.tensor(seg[None, :, :, :, :]),
                        scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                        mode="trilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .numpy()
                )

        # Apply transformations
        transform_result = self.transforms({"image": img, "label": seg})
        
        if isinstance(transform_result, list):
            trans_dict = transform_result[0]
        else:
            trans_dict = transform_result
        
        img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        seg_aug = seg_aug.squeeze()

        img_aug = img_aug.repeat(3, 1, 1, 1)

        if isinstance(self.mask_type, dict):
            mask_type_idx = self.mask_type[idx]
        elif isinstance(self.mask_type, (list, np.ndarray)) and idx < len(self.mask_type):
            mask_type_idx = self.mask_type[idx]
        elif isinstance(self.mask_type, str):
            mask_type_idx = 0 if self.mask_type == "tumor" else 1
        else:
            raise ValueError(f"Invalid mask type: {self.mask_type}")

        return img_aug, seg_aug, np.array(img_vol.header.get_zooms())[self.spatial_index], mask_type_idx

    def get_transforms(self):
        transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[0],
                b_max=self.intensity_range[1],
                clip=True,
            ),
        ]

        if self.split == "train":
            transforms.extend(
                [
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=20,
                        prob=0.5,
                    ),
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0],
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                ]
            )

            if self.do_dummy_2D:
                transforms.extend(
                    [
                       RandRotated(
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            keep_size=False,
                                ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=[1, 0.9, 0.9],
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )
            else:
                transforms.extend(
                    [
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=0.85,
                            max_zoom=1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    binarizeLabeld(keys=["label"]),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),

                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                ]
            )
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    binarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.split == "val"):
            transforms.extend(
                [
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),

                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    binarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    binarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms