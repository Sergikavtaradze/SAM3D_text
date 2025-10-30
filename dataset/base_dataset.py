import pickle
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    AddChanneld,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
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
        print(f"This is the image shape: {img.shape}")
        print(f"This is the seg shape: {seg.shape}")
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
                        # RandRotated(
                        #     keys=["image", "label"],
                        #     prob=0.3,
                        #     range_x=30 / 180 * np.pi,
                        #     range_y=30 / 180 * np.pi,
                        #     range_z=30 / 180 * np.pi,
                        #     keep_size=False,
                        # ),
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
                    # CenterSpatialCropd(
                    #     keys=["image", "label"],
                    #     roi_size=self.rand_crop_spatial_size,
                    # ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                    # RandShiftIntensityd(
                    #     keys=["image"],
                    #     offsets=0.10,
                    #     prob=0.2,
                    # ),
                    # RandGaussianNoised(keys=["image"], prob=0.1),
                    # RandGaussianSmoothd(
                    #     keys=["image"],
                    #     prob=0.2,
                    #     sigma_x=(0.5, 1),
                    #     sigma_y=(0.5, 1),
                    #     sigma_z=(0.5, 1),
                    # ),
                    # AddChanneld(keys=["image", "label"]),
                    # RandShiftIntensityd(keys=["image"], offsets=10),
                    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),]
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
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
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

                    # CenterSpatialCropd(
                    #     keys=["image", "label"],
                    #     roi_size=[32 * (self.rand_crop_spatial_size[i] // 32) for i in range(3)]
                    # ),

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
                        # RandRotated(
                        #     keys=["image", "label"],
                        #     prob=0.3,
                        #     range_x=30 / 180 * np.pi,
                        #     range_y=30 / 180 * np.pi,
                        #     range_z=30 / 180 * np.pi,
                        #     keep_size=False,
                        # ),
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
                    # CenterSpatialCropd(
                    #     keys=["image", "label"],
                    #     roi_size=self.rand_crop_spatial_size,
                    # ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                    # RandShiftIntensityd(
                    #     keys=["image"],
                    #     offsets=0.10,
                    #     prob=0.2,
                    # ),
                    # RandGaussianNoised(keys=["image"], prob=0.1),
                    # RandGaussianSmoothd(
                    #     keys=["image"],
                    #     prob=0.2,
                    #     sigma_x=(0.5, 1),
                    #     sigma_y=(0.5, 1),
                    #     sigma_z=(0.5, 1),
                    # ),
                    # AddChanneld(keys=["image", "label"]),
                    # RandShiftIntensityd(keys=["image"], offsets=10),
                    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),]
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
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
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

                    # CenterSpatialCropd(
                    #     keys=["image", "label"],
                    #     roi_size=[32 * (self.rand_crop_spatial_size[i] // 32) for i in range(3)]
                    # ),

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

    # def visualize_masks(self, idx, pred_mask, output_dir="visualizations"):
    #     """
    #     Visualize the image with overlaid tumor and predicted masks and save to a folder.
        
    #     Args:
    #         idx: Index of the dataset item to visualize
    #         pred_mask: Predicted segmentation mask from the model (numpy array)
    #         output_dir: Directory to save visualizations to
    #     """
    #     import matplotlib.pyplot as plt
    #     import os
    #     import nibabel as nib
    #     import numpy as np
        
    #     # Create output directory if it doesn't exist
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Get the image and mask paths
    #     img_rel_path = self.img_dict[idx]
    #     img_path = os.path.join(self.data_dir, img_rel_path)
        
    #     # Extract patient ID from the file path for naming
    #     patient_id = os.path.basename(img_rel_path).split('.')[0]
        
    #     mask_paths_rel = self.label_dict[idx]
    #     if isinstance(mask_paths_rel, str):
    #         mask_paths_rel = [mask_paths_rel]
    #     mask_paths = [os.path.join(self.data_dir, mp) for mp in mask_paths_rel]
        
    #     # Load the volumes
    #     img_vol = nib.load(img_path)
    #     img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        
    #     tumor_vol = nib.load(mask_paths[0])
    #     tumor = tumor_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        
    #     # Load ICA mask but we won't display it (keeping for return value consistency)
    #     ica = None
    #     if len(mask_paths) > 1:
    #         ica_vol = nib.load(mask_paths[1])
    #         ica = ica_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
    #     else:
    #         ica = np.zeros_like(tumor)
        
    #     # Ensure pred_mask is in the same orientation and format
    #     if pred_mask.shape != img.shape:
    #         print(f"Warning: Predicted mask shape {pred_mask.shape} does not match image shape {img.shape}")
    #         # Attempt to resize the prediction to match the image
    #         try:
    #             from scipy.ndimage import zoom
    #             resize_factors = [img.shape[i] / pred_mask.shape[i] for i in range(3)]
    #             pred_mask = zoom(pred_mask, resize_factors, order=0)  # order=0 for nearest neighbor
    #             print(f"Resized prediction to {pred_mask.shape}")
    #         except Exception as e:
    #             print(f"Failed to resize prediction: {e}")
    #             return img, tumor, ica, pred_mask
        
    #     # Replace NaNs with 0
    #     img[np.isnan(img)] = 0
    #     tumor[np.isnan(tumor)] = 0
    #     pred_mask[np.isnan(pred_mask)] = 0
        
    #     # Find slice with maximum combined mask content (GT + prediction)
    #     # Calculate overlap and combined area for each Z slice
    #     overlap_z = np.sum(np.logical_and(tumor > 0, pred_mask > 0), axis=(1, 2))
    #     combined_z = np.sum(np.logical_or(tumor > 0, pred_mask > 0), axis=(1, 2))
        
    #     # First try to find the slice with maximum overlap
    #     if np.max(overlap_z) > 0:
    #         best_z = np.argmax(overlap_z)
    #     else:
    #         # If no overlap, use the slice with maximum combined area
    #         best_z = np.argmax(combined_z)
        
    #     # Find slices with maximum mask area for tumor and predicted mask
    #     tumor_sum_z = np.sum(tumor, axis=(1, 2))
    #     tumor_best_z = np.argmax(tumor_sum_z)
        
    #     pred_sum_z = np.sum(pred_mask, axis=(1, 2))
    #     pred_best_z = np.argmax(pred_sum_z)
        
    #     # Create the figure with 3 subplots (showing only the best Z slice)
    #     fig, axes = plt.subplots(2, 3, figsize=(18, 6))
    
    #     # 1. Image with ground truth tumor overlay
    #     axes[0, 0].imshow(img[best_z, :, :], cmap='gray')
    #     axes[0, 0].imshow(tumor[best_z, :, :], cmap='hot', alpha=0.5)
    #     axes[0, 0].set_title(f'Ground Truth Mask - Axial (Z={best_z})')
    #     axes[0, 0].axis('off')  # Turn off axis for each subplot individually

    #     # 2. Image with predicted mask overlay
    #     axes[0, 1].imshow(img[best_z, :, :], cmap='gray')
    #     axes[0, 1].imshow(pred_mask[best_z, :, :], cmap='winter', alpha=0.5)
    #     axes[0, 1].set_title(f'Predicted Mask - Axial (Z={best_z})')
    #     axes[0, 1].axis('off')  # Turn off axis for each subplot individually
        
    #     # 3. Image with both GT and predicted mask
    #     # Create a composite image where:
    #     # - GT mask will be green
    #     # - Predicted mask will be blue
    #     # - Overlap will appear cyan (blue+green)
    #     overlay_img = img[best_z, :, :]
    #     # Normalize for better visualization
    #     if overlay_img.max() > overlay_img.min():
    #         overlay_img = (overlay_img - overlay_img.min()) / (overlay_img.max() - overlay_img.min())
        
    #     # Create RGB image
    #     rgb_overlay = np.zeros((*overlay_img.shape, 3))
    #     # Set grayscale background
    #     rgb_overlay[:, :, 0] = overlay_img
    #     rgb_overlay[:, :, 1] = overlay_img 
    #     rgb_overlay[:, :, 2] = overlay_img
        
    #     # Add GT mask in green
    #     rgb_overlay[:, :, 1] = np.where(tumor[best_z, :, :] > 0, 1.0, rgb_overlay[:, :, 1])
        
    #     # Add predicted mask in blue (not red)
    #     rgb_overlay[:, :, 2] = np.where(pred_mask[best_z, :, :] > 0, 1.0, rgb_overlay[:, :, 2])
        
    #     axes[0, 2].imshow(rgb_overlay)
    #     axes[0, 2].set_title(f'GT (green) & Pred (blue) - Axial (Z={best_z})')
    #     axes[0, 2].axis('off')  # Turn off axis for each subplot individually

    #     # 4. ground truth tumor mask
    #     axes[1, 0].imshow(tumor[best_z, :, :], cmap='hot', alpha=0.5)
    #     axes[1, 0].set_title(f'Ground Truth Mask - Axial (Z={best_z})')
    #     axes[1, 0].axis('off')  # Turn off axis for each subplot individually
    #     # 5. predicted mask
    #     axes[1, 1].imshow(pred_mask[best_z, :, :], cmap='winter', alpha=0.5)
    #     axes[1, 1].set_title(f'Predicted Mask - Axial (Z={best_z})')
    #     axes[1, 1].axis('off')  # Turn off axis for each subplot individually

    #     # 6. predicted mask with ground truth tumor mask overlay without image
    #     axes[1, 2].imshow(pred_mask[best_z, :, :], cmap='winter', alpha=0.5)
    #     axes[1, 2].imshow(tumor[best_z, :, :], cmap='hot', alpha=0.5)
    #     axes[1, 2].set_title(f'Predicted Mask with GT Tumor Mask - Axial (Z={best_z})')
    #     axes[1, 2].axis('off')  # Turn off axis for each subplot individually
        
        
    #     plt.tight_layout()
    #     fig_path = os.path.join(output_dir, f"{patient_id}_idx{idx}_comparison.png")
    #     plt.savefig(fig_path, dpi=150)
    #     plt.close()
        
    #     # Create a second figure showing the max slices for better visualization
    #     fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
        
    #     # Tumor maximum slice
    #     axes2[0, 0].imshow(img[tumor_best_z, :, :], cmap='gray')
    #     axes2[0, 0].set_title(f'Image - Axial (Tumor Max Z={tumor_best_z})')
        
    #     axes2[0, 1].imshow(img[tumor_best_z, :, :], cmap='gray')
    #     axes2[0, 1].imshow(tumor[tumor_best_z, :, :], cmap='hot', alpha=0.5)
    #     axes2[0, 1].set_title(f'Tumor Mask - Axial (Max Z={tumor_best_z})')
        
    #     # Predicted mask maximum slice
    #     axes2[1, 0].imshow(img[pred_best_z, :, :], cmap='gray')
    #     axes2[1, 0].set_title(f'Image - Axial (Pred Max Z={pred_best_z})')
        
    #     axes2[1, 1].imshow(img[pred_best_z, :, :], cmap='gray')
    #     axes2[1, 1].imshow(pred_mask[pred_best_z, :, :], cmap='winter', alpha=0.5)  # Using blue colormap
    #     axes2[1, 1].set_title(f'Pred Mask - Axial (Max Z={pred_best_z})')
        
    #     # Turn off axes for all subplots
    #     for ax in axes2.flat:
    #         ax.axis('off')
        
    #     plt.tight_layout()
    #     fig2_path = os.path.join(output_dir, f"{patient_id}_idx{idx}_max_slices.png")
    #     plt.savefig(fig2_path, dpi=150)
    #     plt.close()
        
    #     # Save statistics to a text file
    #     stats_path = os.path.join(output_dir, f"{patient_id}_idx{idx}_stats.txt")
    #     with open(stats_path, 'w') as f:
    #         f.write(f"Image shape: {img.shape}\n")
    #         f.write(f"Tumor mask shape: {tumor.shape}\n")
    #         f.write(f"Predicted mask shape: {pred_mask.shape}\n")
    #         f.write(f"Tumor mask range: [{np.min(tumor)}, {np.max(tumor)}]\n")
    #         f.write(f"Predicted mask range: [{np.min(pred_mask)}, {np.max(pred_mask)}]\n")
    #         f.write(f"Tumor mask positive voxels: {np.sum(tumor > 0)}\n")
    #         f.write(f"Predicted mask positive voxels: {np.sum(pred_mask > 0)}\n")
    #         f.write(f"Overlap (intersection) voxels: {np.sum(np.logical_and(tumor > 0, pred_mask > 0))}\n")
    #         f.write(f"Best Z slice: {best_z}\n")
    #         f.write(f"Tumor best Z slice: {tumor_best_z}\n")
    #         f.write(f"Prediction best Z slice: {pred_best_z}\n")
        
    #     print(f"Visualization saved to {output_dir} for patient {patient_id} (index {idx})")
    #     return img, tumor, ica, pred_mask