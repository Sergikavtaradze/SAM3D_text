from dataset.datasets import load_data_volume_maskwise
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer, PromptEncoder_Text
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
from torch.utils.data import ConcatDataset, DataLoader

from transformers import GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import random

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def seed_everything(seed=24):
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["maskwise_endonasal"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--num_prompts",
        default=1,
        type=int,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument(
        "--checkpoint",
        default="best",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--target_class", default=None, choices=["tumor", "ICA"], type=str)
    parser.add_argument(
        "--pkl_filename",
        default="excluded_ids_maskwise_split_updated_124MRIs_combined_val_test.pkl",
        type=str,
    )
    parser.add_argument("--seed", default='24', type=str)

    args = parser.parse_args()

    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["maskwise_endonasal"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    #args.snapshot_path = os.path.join(args.snapshot_path)

    setup_logger(logger_name="test", root='', screen=True, tofile=False)
    logger = logging.getLogger(f"test")
    logger.info(str(args))

    # Load validation data
    val_data = load_data_volume_maskwise(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        pkl_filename=args.pkl_filename,
        augmentation=False,
        split="val",
        target_class=args.target_class,
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=0
    )

    # Load test data
    test_data = load_data_volume_maskwise(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        pkl_filename=args.pkl_filename,
        augmentation=False,
        split="test",
        target_class=args.target_class,
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=0
    )

    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)
    img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu', weights_only=False)["encoder_dict"], strict=True)
    img_encoder.to(device)

    prompt_encoder_text = PromptEncoder_Text().to(device)
    state = torch.load(os.path.join(args.snapshot_path, file), map_location='cpu', weights_only=False)
    feature_dict = state["feature_dict"]
    if isinstance(feature_dict, list):
        state_dict = feature_dict[3]
    else:
        state_dict = feature_dict
    prompt_encoder_text.load_state_dict(state_dict, strict=True)
    prompt_encoder_text.eval()
    mask_decoder = VIT_MLAHead(img_size = 96, num_classes=2).to(device)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu', weights_only=False)["decoder_dict"],
                          strict=True)
    mask_decoder.to(device)

    dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")
    img_encoder.eval()
    mask_decoder.eval()

    patch_size = args.rand_crop_size[0]

    def model_predict(img, img_encoder, mask_decoder):#, prompt): For now we dont input the prompt
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)
        #feature_list = feature_list[::-1]
        new_feature = []
        for i, feature in enumerate(feature_list):#, prompt_encoder)):
            if i == 3:
                ############################################################################
                if args.target_class == "tumor":
                    prompt_text = "Pituitary Tumor"
                elif args.target_class == "ICA":
                    prompt_text = "Internal Carotid Artery"
                elif args.target_class is None:
                    raise ValueError("Target class is not specified, please specify ICA or tumor")
                text_features = prompt_encoder_text(feature, prompt_text)
                new_feature.append(text_features)
            else:
                new_feature.append(feature.to(device))

        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size,
                                   mode="trilinear")
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, patch_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        return masks

    def save_prediction_to_mri(args, predictions, patient_ids, seed):
        """Save the predictions as NIFTI (.nii.gz) files."""
        #logging.info("inside save_prediction_to_mri function")
        # Create the predicted_segs directory inside the snapshot_path
        save_dir = os.path.join(args.snapshot_path, 'predicted_segs')
        array_dir = os.path.join(args.snapshot_path, 'seg_arrays')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(array_dir, exist_ok=True)
        #logging.info("got to makedirs")
        for idx, patient_id in enumerate(patient_ids):
            #logging.info("Got into the for loop")
            path = os.path.join(args.data_prefix, patient_id) # Original image path
            img_original = nib.load(path) # Load the original image
            
            # img_shape = img_original.get_fdata().shape
            prediction = predictions[idx].astype(np.float32)
            prediction = prediction.transpose(2, 1, 0)
            
            # np.save(array_dir + "/" + "pred_tumor.npy", predictions[idx])
            # np.save(array_dir + "/" + "img_original.npy", img_original.get_fdata())

            try:
                img_nifti = nib.Nifti1Image(prediction, img_original.affine, header=img_original.header)
                #logging.info("got to img_nifti")
            except Exception as e:
                logging.error(f"Error creating NIFTI image: {e}")
                continue
            # Save the predicted mask as a NIFTI file
            if args.target_class == "tumor":
                os.makedirs(save_dir + f"/tumor_{seed}", exist_ok=True)
                nib.save(img_nifti, os.path.join(save_dir+f'/tumor_{seed}/mri085_pred.nii.gz'))
                logging.info(f"Saved prediction for {patient_id} to {save_dir+f'/tumor_{seed}/mri085_pred.nii.gz'}")
            
            elif args.target_class == "ICA":
                os.makedirs(save_dir + f"/ICA_{seed}", exist_ok=True)
                nib.save(img_nifti, os.path.join(save_dir+f'/ICA_{seed}/mri085_pred.nii.gz'))
                logging.info(f"Saved prediction for {patient_id} to {save_dir+f'/ICA_{seed}/mri085_pred.nii.gz'}")


    with torch.no_grad():
        # Separate metrics for validation and test
        val_loss_summary = []
        val_loss_nsd = []
        val_case_metrics = []
        
        test_loss_summary = []
        test_loss_nsd = []
        test_case_metrics = []
        
        # Process validation data
        logger.info("Processing validation data...")
        for idx, (img, seg, spacing, mask_type_idx) in enumerate(val_data):
            seg = seg[:, 1, :, :, :] # No need for the background mask
            
            seg = seg.float()
            prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)
            seg_pred = torch.zeros_like(prompt).to(device)
            l = len(torch.where(prompt == 1)[0])
            sample = np.random.choice(np.arange(l), args.num_prompts, replace=True)
            x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
            y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
            z = torch.where(prompt == 1)[2][sample].unsqueeze(1)

            x_m = (torch.max(x) + torch.min(x)) // 2
            y_m = (torch.max(y) + torch.min(y)) // 2
            z_m = (torch.max(z) + torch.min(z)) // 2

            d_min = x_m - patch_size//2
            d_max = x_m + patch_size//2
            h_min = z_m - patch_size//2
            h_max = z_m + patch_size//2
            w_min = y_m - patch_size//2
            w_max = y_m + patch_size//2
            d_l = max(0, -d_min)
            d_r = max(0, d_max - prompt.shape[1])
            h_l = max(0, -h_min)
            h_r = max(0, h_max - prompt.shape[2])
            w_l = max(0, -w_min)
            w_r = max(0, w_max - prompt.shape[3])
            
            d_min = max(0, d_min)
            h_min = max(0, h_min)
            w_min = max(0, w_min)
            img_patch = img[:, :, d_min:d_max, h_min:h_max, w_min:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
            
            # Run inference with text prompts
            pred = model_predict(img_patch, img_encoder, mask_decoder)
            
            pred = pred[:,:, d_l:patch_size-d_r, h_l:patch_size-h_r, w_l:patch_size-w_r]
            pred = F.softmax(pred, dim=1)[:,1]
            seg_pred[:, d_min:d_max, h_min:h_max, w_min:w_max] += pred

            final_pred = F.interpolate(seg_pred.unsqueeze(1), size = seg.shape[2:], mode="trilinear")
            masks = final_pred > 0.5
            loss = 1 - dice_loss(masks, seg)
            val_loss_summary.append(loss.detach().cpu().numpy())

            ssd = surface_distance.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(),
                                                             (masks==1)[0, 0].cpu().numpy(),
                                                             spacing_mm=spacing[0].numpy())
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)
            val_loss_nsd.append(nsd)
            
            # Store case info for visualization
            val_case_metrics.append({
                'idx': idx,
                'name': val_data.dataset.img_dict[idx],
                'dice': loss.item(),
                'nsd': nsd,
                'image': img[0, 0].cpu().numpy(),
                'ground_truth': seg[0, 0].cpu().numpy(),
                'prediction': masks[0, 0].cpu().numpy()
            })
            
            logger.info(
                " Validation Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    val_data.dataset.img_dict[idx], loss.item(), nsd
                ))

        # Process test data
        logger.info("Processing test data...")
        for idx, (img, seg, spacing, mask_type_idx) in enumerate(test_data):

            seg = seg[:, 1, :, :, :] # No need for the background mask
            # Log mask type for clarity
            
            seg = seg.float()
            prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)
            seg_pred = torch.zeros_like(prompt).to(device)
            l = len(torch.where(prompt == 1)[0])
            sample = np.random.choice(np.arange(l), args.num_prompts, replace=True)
            x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
            y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
            z = torch.where(prompt == 1)[2][sample].unsqueeze(1)

            x_m = (torch.max(x) + torch.min(x)) // 2
            y_m = (torch.max(y) + torch.min(y)) // 2
            z_m = (torch.max(z) + torch.min(z)) // 2

            d_min = x_m - patch_size//2
            d_max = x_m + patch_size//2
            h_min = z_m - patch_size//2
            h_max = z_m + patch_size//2
            w_min = y_m - patch_size//2
            w_max = y_m + patch_size//2
            d_l = max(0, -d_min)
            d_r = max(0, d_max - prompt.shape[1])
            h_l = max(0, -h_min)
            h_r = max(0, h_max - prompt.shape[2])
            w_l = max(0, -w_min)
            w_r = max(0, w_max - prompt.shape[3])
            
            d_min = max(0, d_min)
            h_min = max(0, h_min)
            w_min = max(0, w_min)
            img_patch = img[:, :, d_min:d_max, h_min:h_max, w_min:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))

            # print(f"image x idxs: {d_min}, {d_max}") 
            # print(f"image y idxs: {h_min}, {h_max}")
            # print(f"image z idxs: {w_min}, {w_max}")
            
            # Run inference with text prompts
            pred = model_predict(img_patch, img_encoder, mask_decoder)
            
            pred = pred[:,:, d_l:patch_size-d_r, h_l:patch_size-h_r, w_l:patch_size-w_r]
            pred = F.softmax(pred, dim=1)[:,1]
            seg_pred[:, d_min:d_max, h_min:h_max, w_min:w_max] += pred

            final_pred = F.interpolate(seg_pred.unsqueeze(1), size = seg.shape[2:], mode="trilinear")
            masks = final_pred > 0.5
            loss = 1 - dice_loss(masks, seg)
            test_loss_summary.append(loss.detach().cpu().numpy())

            ssd = surface_distance.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(),
                                                             (masks==1)[0, 0].cpu().numpy(),
                                                             spacing_mm=spacing[0].numpy())
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)
            test_loss_nsd.append(nsd)
            
            # Store case info for visualization
            test_case_metrics.append({
                'idx': idx,
                'name': test_data.dataset.img_dict[idx],
                'dice': loss.item(),
                'nsd': nsd,
                'image': img[0, 0].cpu().numpy(),
                'ground_truth': seg[0, 0].cpu().numpy(),
                'prediction': masks[0, 0].cpu().numpy()
            })
            
            logger.info(
                " Test Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    test_data.dataset.img_dict[idx], loss.item(), nsd
                ))

        # Combine validation and test metrics
        combined_loss_summary = val_loss_summary + test_loss_summary
        combined_loss_nsd = val_loss_nsd + test_loss_nsd
        combined_case_metrics = val_case_metrics + test_case_metrics
        
        # Log individual dataset metrics
        logger.info("=" * 50)
        logger.info("VALIDATION METRICS:")
        logger.info(f"- Validation samples: {len(val_loss_summary)}")
        if len(val_loss_summary) > 0:
            logger.info(f"- Validation Dice: {np.mean(val_loss_summary):.6f} ± {np.std(val_loss_summary):.6f}")
            logger.info(f"- Validation NSD: {np.mean(val_loss_nsd):.6f} ± {np.std(val_loss_nsd):.6f}")
        else:
            logger.info("- No validation data available")
            
        logger.info("=" * 50)
        logger.info("TEST METRICS:")
        logger.info(f"- Test samples: {len(test_loss_summary)}")
        if len(test_loss_summary) > 0:
            logger.info(f"- Test Dice: {np.mean(test_loss_summary):.6f} ± {np.std(test_loss_summary):.6f}")
            logger.info(f"- Test NSD: {np.mean(test_loss_nsd):.6f} ± {np.std(test_loss_nsd):.6f}")
        else:
            logger.info("- No test data available")
            
        logger.info("=" * 50)
        logger.info("COMBINED METRICS (VALIDATION + TEST):")
        logger.info(f"- Total samples: {len(combined_loss_summary)}")
        if len(combined_loss_summary) > 0:
            logger.info(f"- Combined Dice: {np.mean(combined_loss_summary):.6f} ± {np.std(combined_loss_summary):.6f}")
            logger.info(f"- Combined NSD: {np.mean(combined_loss_nsd):.6f} ± {np.std(combined_loss_nsd):.6f}")
        else:
            logger.info("- No data available")
        logger.info("=" * 50)
        
        # Save predictions as NIFTI files
        #patient_ids = [data['name'] for data in case_metrics]
        #predictions = [data['prediction'].astype(np.uint8) for data in case_metrics]
        
        # save_prediction_to_mri(args, predictions, patient_ids, args.seed)

        # Sort all cases by Dice score (highest to lowest)
        # case_metrics.sort(key=lambda x: x['dice'], reverse=True)
        
        # Visualize all cases from best to worst
        # visualize_all_cases(case_metrics, args.snapshot_path, args.target_class, args.seed)


def visualize_all_cases(case_metrics, save_path, target_class, seed):
    """Visualize all test cases sorted from best to worst based on Dice score."""
    # Create visualization directory if it doesn't exist
    if target_class == "tumor":
        save_dir = os.path.join(save_path, f'visualizations_tumor_MaxPredDensity_{seed}')
    elif target_class == "ICA":
        save_dir = os.path.join(save_path, f'visualizations_ICA_MaxPredDensity_{seed}')
    else:
        raise ValueError("Target class is not specified, please specify ICA or tumor")
    
    os.makedirs(save_dir, exist_ok=True)
    
    total_cases = len(case_metrics)
    
    for rank, case_data in enumerate(case_metrics, 1):
        # Extract data
        gt = case_data['ground_truth']
        pred = case_data['prediction']
        img = case_data['image']
        
        # Count the number of predicted foreground voxels (where pred > 0) for each axial slice (Z-axis).
        # Results in a 1D array where each element corresponds to the count of foreground voxels in a specific depth slice.
        foreground_per_z_slice = np.sum(pred > 0, axis=(0, 1))
        if np.max(foreground_per_z_slice) == 0:
            foreground_per_z_slice = np.sum(gt > 0, axis=(0, 1)) # Fallback to GT
            if np.max(foreground_per_z_slice) == 0:
                mid_z = img.shape[2] // 2 # Default to center if both are empty
            else:
                mid_z = np.argmax(foreground_per_z_slice)
        else:
            mid_z = np.argmax(foreground_per_z_slice)
        
        # Count the number of predicted foreground voxels (where pred > 0) for each coronal slice (Y-axis).
        # Results in 1D array where each element corresponds to the count of foreground voxels in a specific coronal slice.
        foreground_per_y_slice = np.sum(pred > 0, axis=(0, 2))
        if np.max(foreground_per_y_slice) == 0:
            foreground_per_y_slice = np.sum(gt > 0, axis=(0, 2)) # Fallback to GT
            if np.max(foreground_per_y_slice) == 0:
                mid_y = img.shape[1] // 2 # Default to center if both are empty
            else:
                mid_y = np.argmax(foreground_per_y_slice)
        else:
            mid_y = np.argmax(foreground_per_y_slice)
        
        # Create the figure with axial and coronal views (2 rows, 6 columns)
        plt.figure(figsize=(18, 18))
        
        # ===== AXIAL VIEWS (TOP ROW) =====
        
        # Row 1, Column 1: Ground Truth Mask on MRI - Axial
        plt.subplot(4, 3, 1)
        plt.title(f"Ground Truth Mask - Axial (Z={mid_z})")
        plt.imshow(img[:, :, mid_z], cmap='gray')
        gt_mask = np.ma.masked_where(gt[:, :, mid_z] < 0.5, gt[:, :, mid_z])
        plt.imshow(gt_mask, alpha=0.7, cmap='Greens')
        plt.axis('off')
        
        # Row 1, Column 2: Predicted Mask on MRI - Axial
        plt.subplot(4, 3, 2)
        plt.title(f"Predicted Mask - Axial (Z={mid_z})")
        plt.imshow(img[:, :, mid_z], cmap='gray')
        pred_mask = np.ma.masked_where(pred[:, :, mid_z] < 0.5, pred[:, :, mid_z])
        plt.imshow(pred_mask, alpha=0.7, cmap='Blues')
        plt.axis('off')
        
        # Row 1, Column 3: GT and Pred overlay on MRI - Axial
        plt.subplot(4, 3, 3)
        plt.title(f"GT (green) & Pred (blue) - Axial (Z={mid_z})")
        plt.imshow(img[:, :, mid_z], cmap='gray')
        gt_mask = np.ma.masked_where(gt[:, :, mid_z] < 0.5, gt[:, :, mid_z])
        pred_mask = np.ma.masked_where(pred[:, :, mid_z] < 0.5, pred[:, :, mid_z])
        plt.imshow(gt_mask, alpha=0.5, cmap='Greens')
        plt.imshow(pred_mask, alpha=0.5, cmap='Blues')
        plt.axis('off')
        
        # Row 2, Column 1: Ground Truth Mask only - Axial
        plt.subplot(4, 3, 4)
        plt.title(f"Ground Truth Mask - Axial (Z={mid_z})")
        plt.imshow(np.ones_like(gt[:, :, mid_z]) * 0.7, cmap='gray')  # Gray background
        plt.imshow(gt[:, :, mid_z], cmap='gray')  # White mask
        plt.axis('off')
        
        # Row 2, Column 2: Predicted Mask only - Axial
        plt.subplot(4, 3, 5)
        plt.title(f"Predicted Mask - Axial (Z={mid_z})")
        plt.imshow(np.ones_like(pred[:, :, mid_z]) * 0.7, cmap='Blues')  # Blue tinted background
        plt.imshow(pred[:, :, mid_z], cmap='Blues')  # Blue mask
        plt.axis('off')
        
        # Row 2, Column 3: Predicted Mask with GT overlay - Axial
        plt.subplot(4, 3, 6)
        plt.title(f"Predicted Mask with GT Tumor Mask - Axial (Z={mid_z})")
        plt.imshow(np.ones_like(pred[:, :, mid_z]) * 0.7, cmap='Blues')  # Blue tinted background
        plt.imshow(pred[:, :, mid_z], cmap='Blues')  # Blue prediction mask
        gt_mask = np.ma.masked_where(gt[:, :, mid_z] < 0.5, gt[:, :, mid_z])
        plt.imshow(gt_mask, alpha=0.7, cmap='gray')  # White GT overlay
        plt.axis('off')
        
        # ===== CORONAL VIEWS (BOTTOM ROW) =====
        # For coronal view, we need to slice along mid_y
        # We transpose and flip to keep anatomical orientation correct
        
        # Row 3, Column 1: Ground Truth Mask on MRI - Coronal
        plt.subplot(4, 3, 7)
        plt.title(f"Ground Truth Mask - Coronal (Y={mid_y})")
        # Display the coronal slice of the image
        coronal_img = img[:, mid_y, :].T
        plt.imshow(coronal_img, cmap='gray')
        # Create mask for the ground truth
        coronal_gt = gt[:, mid_y, :].T
        gt_mask = np.ma.masked_where(coronal_gt < 0.5, coronal_gt)
        plt.imshow(gt_mask, alpha=0.7, cmap='Greens')
        plt.axis('off')
        
        # Row 3, Column 2: Predicted Mask on MRI - Coronal
        plt.subplot(4, 3, 8)
        plt.title(f"Predicted Mask - Coronal (Y={mid_y})")
        plt.imshow(coronal_img, cmap='gray')
        # Create mask for the prediction
        coronal_pred = pred[:, mid_y, :].T
        pred_mask = np.ma.masked_where(coronal_pred < 0.5, coronal_pred)
        plt.imshow(pred_mask, alpha=0.7, cmap='Blues')
        plt.axis('off')
        
        # Row 3, Column 3: GT and Pred overlay on MRI - Coronal
        plt.subplot(4, 3, 9)
        plt.title(f"GT (green) & Pred (blue) - Coronal (Y={mid_y})")
        plt.imshow(coronal_img, cmap='gray')
        gt_mask = np.ma.masked_where(coronal_gt < 0.5, coronal_gt)
        pred_mask = np.ma.masked_where(coronal_pred < 0.5, coronal_pred)
        plt.imshow(gt_mask, alpha=0.5, cmap='Greens')
        plt.imshow(pred_mask, alpha=0.5, cmap='Blues')
        plt.axis('off')
        
        # Row 4, Column 1: Ground Truth Mask only - Coronal
        plt.subplot(4, 3, 10)
        plt.title(f"Ground Truth Mask - Coronal (Y={mid_y})")
        plt.imshow(np.ones_like(coronal_gt) * 0.7, cmap='gray')  # Gray background
        plt.imshow(coronal_gt, cmap='gray')  # White mask
        plt.axis('off')
        
        # Row 4, Column 2: Predicted Mask only - Coronal
        plt.subplot(4, 3, 11)
        plt.title(f"Predicted Mask - Coronal (Y={mid_y})")
        plt.imshow(np.ones_like(coronal_pred) * 0.7, cmap='Blues')  # Blue tinted background
        plt.imshow(coronal_pred, cmap='Blues')  # Blue mask
        plt.axis('off')
        
        # Row 4, Column 3: Predicted Mask with GT overlay - Coronal
        plt.subplot(4, 3, 12)
        plt.title(f"Predicted Mask with GT Tumor Mask - Coronal (Y={mid_y})")
        plt.imshow(np.ones_like(coronal_pred) * 0.7, cmap='Blues')  # Blue tinted background
        plt.imshow(coronal_pred, cmap='Blues')  # Blue prediction mask
        gt_mask = np.ma.masked_where(coronal_gt < 0.5, coronal_gt)
        plt.imshow(gt_mask, alpha=0.7, cmap='gray')  # White GT overlay
        plt.axis('off')
        
        # Add case information as a small text at the bottom
        plt.figtext(0.5, 0.01, 
                  f"Case {rank}/{total_cases}: {case_data['name']} | Dice: {case_data['dice']:.4f}, NSD: {case_data['nsd']:.4f}", 
                  ha='center', fontsize=10)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Pad rank with zeros for proper sorting in file browsers
        padded_rank = str(rank).zfill(len(str(total_cases)))
        
        # Save figure
        sanitized_name = str(case_data['name']).replace('/', '_').replace('\\', '_').replace(' ', '_')
        plt.savefig(os.path.join(save_dir, f"case_{padded_rank}_dice_{case_data['dice']:.4f}_{sanitized_name}.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        if rank % 10 == 0:  # Print progress every 10 cases
            logging.info(f"Visualized {rank}/{total_cases} cases")
    
    logging.info(f"All {total_cases} visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()

