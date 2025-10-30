from dataset.datasets import load_data_volume, load_data_volume_maskwise
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer, PromptEncoder_Text_Llama
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger
import random

def seed_everything(seed=24):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon", "endonasal", "maskwise_endonasal"]
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
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--date", required=True, type=str)
    parser.add_argument(
        "--llama_model", 
        default="meta-llama/llama-3.2-1b", 
        type=str,
        help="LLAMA model variant to use (e.g., meta-llama/llama-3.2-1b, meta-llama/llama-3.2-3b)"
    )

    args = parser.parse_args()
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits", "endonasal", "maskwise_endonasal"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    args.snapshot_path = os.path.join(args.snapshot_path, args.data, 
    "MaskWise_Excluded_IDs_LLAMA_" + "lr_" + str(args.lr) + "crop" + str(args.rand_crop_size) + args.date)
    
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    
    setup_logger(logger_name="train", root='', screen=True, tofile=False)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    logger.info(f"Using LLAMA model: {args.llama_model}")

    train_data = load_data_volume_maskwise(
        data=args.data,
        path_prefix=args.data_prefix,
        pkl_filename="excluded_ids_maskwise_split.pkl", # maskwise_split.pkl is another option
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )
    
    val_data = load_data_volume_maskwise(
        data=args.data,
        path_prefix=args.data_prefix,
        pkl_filename="excluded_ids_maskwise_split.pkl",
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )

    #print the train_data and val_data on Colab to check their output, shape, etc.

    #sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    
    # Build base model
    sam, img_embedding_size = sam_model_registry["vit_b"](
        image_size=1024,  # Standard SAM image size
        # I attempted different image sizes, but image size of 1024
        # gives the leas amount of size mismatches.
        # Now there are size mismatches between torch.Size
        # torch.Size([4, 256]) from checkpoint, the shape in current model is torch.Size([3, 256])
        num_classes=3, #args.num_classes,  # Use num_classes from args
        # Worked when changing the number of classes to 3 
        checkpoint=None # Initialize without checkpoint
    )

    # Save the checkpoint path
    checkpoint_path = os.path.join("ckpt", "sam_vit_b_01ec64.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}. Please ensure the checkpoint file exists.")

    # Load the checkpoint
    sam.load_state_dict(torch.load(checkpoint_path))

    # Move to specified device
    sam = sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    # Create a new 3D image encoder with specific architecture parameters:
    img_encoder = ImageEncoderViT_3d(
        depth=12,                   # Number of transformer layers
        embed_dim=768,              # Dimension of embedded features
        img_size=1024,              # Input image size
        mlp_ratio=4,                # Ratio for MLP layer dimensions
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # Normalization layer with epsilon
        num_heads=12,               # Number of attention heads
        patch_size=16,              # Size of image patches
        qkv_bias=True,              # Use bias in query, key, value projections
        use_rel_pos=True,           # Use relative positional embeddings
        global_attn_indexes=[2, 5, 8, 11],  # Layers that use global attention
        window_size=14,             # Size of local attention window
        cubic_window_size=8,        # Size of 3D attention window
        out_chans=256,              # Number of output channels
        num_slice=16                # Number of slices in 3D volume
    )

    # Load the weights from SAM's 2D image encoder into our 3D encoder
    # strict=False allows loading weights even if architectures don't exactly match
    img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)

    # Delete the original SAM model to free up memory
    del sam
    
    # Move the 3D image encoder to the specified device (GPU/CPU)
    img_encoder.to(device)


    ############################################################
    # Freeze all parameters in the image encoder
    # And then unfreeze specific parameters for training
    ############################################################
    for p in img_encoder.parameters():
        p.requires_grad = False
    img_encoder.depth_embed.requires_grad = True
    for p in img_encoder.slice_embed.parameters():
        p.requires_grad = True

    # For each transformer block in the encoder:
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
        i.attn.rel_pos_d = nn.parameter.Parameter(
            0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), 
            requires_grad=True
        )

    # Enable training for all parameters in the 3D neck layers
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = True

    ############################################################
    # LLAMA3.2 Prompt Encoder Setup
    ############################################################
    
    # Only 1 prompt encoder is used in the final layer (index 3)
    prompt_encoder = PromptEncoder_Text_Llama()
    prompt_encoder.to(device)
    parameter_list = [i for i in prompt_encoder.parameters() if i.requires_grad]

    logger.info(f"Total trainable parameters in prompt encoders: {len(parameter_list)}")
    
    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2) # Changed to 3 from 2 and now changed back

    mask_decoder.to(device)

    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    feature_opt = AdamW(parameter_list, lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=500)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    
    dice_loss = DiceLoss(include_background=False, softmax=True, 
                        to_onehot_y=False, reduction="none") 
    
    loss_cal = DiceCELoss(include_background=False, softmax=True, 
                          to_onehot_y=False, lambda_dice=0.5, lambda_ce=0.5)
    best_loss = np.inf
    best_epoch = 0
    patch_size = args.rand_crop_size[0]

    logger.info("Starting training with LLAMA3.2 prompt encoder...")
    
    for epoch_num in range(args.max_epoch):
        loss_summary_train = []
        img_encoder.train()
        prompt_encoder.train()
        mask_decoder.train()
        
        for idx, (img, seg, spacing, mask_type) in enumerate(train_data):
            
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1)
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)
            
            

            new_feature = feature_list.copy()
            # Only process the feature at index 3 with the prompt encoder
            if mask_type == 0:
                text_prompt = "Pituitary Tumor"
            elif mask_type == 1:
                text_prompt = "Internal Carotid Artery"
            else:
                raise ValueError(f"Invalid channel during prompt encoder training: {mask_type}")

            prompt_feature = prompt_encoder(feature_list[3], text_prompt)
            new_feature[3] = prompt_feature

            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                mode='trilinear')
            new_feature.append(img_resize)
            masks = mask_decoder(new_feature, 2, patch_size//64)
            masks = masks.permute(0, 1, 4, 2, 3)
            seg = seg.to(device)
            # seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            loss_summary_train.append(loss.detach().cpu().numpy())
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), 1.0)
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
            
            # Clear cache more frequently for LLAMA models
            if idx % 5 == 0:
                torch.cuda.empty_cache()
        
        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()

        img_encoder.eval()
        prompt_encoder.eval()
        mask_decoder.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, (img, seg, spacing, mask_type) in enumerate(val_data):
                out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
                input_batch = out.to(device)
                input_batch = input_batch[0].transpose(0, 1)
                
                batch_features, feature_list = img_encoder(input_batch)
                feature_list.append(batch_features)
                
                new_feature = feature_list.copy()
                # Only process the feature at index 3 with the prompt encoder
                if mask_type == 0:
                    text_prompt = "Pituitary Tumor"
                elif mask_type == 1:
                    text_prompt = "Internal Carotid Artery"
                else:
                    raise ValueError(f"Invalid channel during prompt encoder validation: {mask_type}")

                prompt_feature = prompt_encoder(feature_list[3], text_prompt)
                new_feature[3] = prompt_feature
                        
                img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64/patch_size,
                                           mode='trilinear')
                new_feature.append(img_resize)
                masks = mask_decoder(new_feature, 2, patch_size//64)
                masks = masks.permute(0, 1, 4, 2, 3)
                seg = seg.to(device)
                # seg = seg.unsqueeze(1)
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                torch.cuda.empty_cache()  # Clear GPU cache after each batch

        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            best_epoch = epoch_num
            is_best = True
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": prompt_encoder.state_dict(),
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict(),
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info(
                'epoch: {}/{}'.format(epoch_num, args.max_epoch) + ": Train Loss:" + str(
                    np.mean(loss_summary_train)) + ", Val Loss:" + str(np.mean(loss_summary)) + ", Best Val Loss:" + str(
                        best_loss) + ", Best Epoch:" + str(best_epoch))


if __name__ == "__main__":
    main()