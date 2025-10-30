def visualize_masks(self, idx, output_dir="visualizations"):
    """
    Visualize the image with overlaid tumor and ICA masks and save to a folder.
    
    Args:
        idx: Index of the dataset item to visualize
        output_dir: Directory to save visualizations to
    """
    import matplotlib.pyplot as plt
    import os
    import nibabel as nib
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the image and mask paths
    img_rel_path = self.img_dict[idx]
    img_path = os.path.join(self.data_dir, img_rel_path)
    
    # Extract patient ID from the file path for naming
    patient_id = os.path.basename(img_rel_path).split('.')[0]
    
    mask_paths_rel = self.label_dict[idx]
    if isinstance(mask_paths_rel, str):
        mask_paths_rel = [mask_paths_rel]
    mask_paths = [os.path.join(self.data_dir, mp) for mp in mask_paths_rel]
    
    # Load the volumes
    img_vol = nib.load(img_path)
    img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
    
    tumor_vol = nib.load(mask_paths[0])
    ica_vol = nib.load(mask_paths[1])
    tumor = tumor_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
    ica = ica_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
    
    # Replace NaNs with 0
    img[np.isnan(img)] = 0
    tumor[np.isnan(tumor)] = 0
    ica[np.isnan(ica)] = 0
    
    # Get the middle slice for each dimension
    slice_z = img.shape[0] // 2
    slice_y = img.shape[1] // 2
    slice_x = img.shape[2] // 2
    
    # Create the figure with 3 subplots (one for each viewing plane)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Axial view (Z plane)
    axes[0, 0].imshow(img[slice_z, :, :], cmap='gray')
    axes[0, 0].set_title('Image - Axial')
    
    axes[0, 1].imshow(img[slice_z, :, :], cmap='gray')
    axes[0, 1].imshow(tumor[slice_z, :, :], cmap='hot', alpha=0.5)
    axes[0, 1].set_title('Tumor Mask - Axial')
    
    axes[0, 2].imshow(img[slice_z, :, :], cmap='gray')
    axes[0, 2].imshow(ica[slice_z, :, :], cmap='winter', alpha=0.5)
    axes[0, 2].set_title('ICA Mask - Axial')
    
    # Coronal view (Y plane)
    axes[1, 0].imshow(img[:, slice_y, :], cmap='gray')
    axes[1, 0].set_title('Image - Coronal')
    
    axes[1, 1].imshow(img[:, slice_y, :], cmap='gray')
    axes[1, 1].imshow(tumor[:, slice_y, :], cmap='hot', alpha=0.5)
    axes[1, 1].set_title('Tumor Mask - Coronal')
    
    axes[1, 2].imshow(img[:, slice_y, :], cmap='gray')
    axes[1, 2].imshow(ica[:, slice_y, :], cmap='winter', alpha=0.5)
    axes[1, 2].set_title('ICA Mask - Coronal')
    
    # Sagittal view (X plane)
    axes[2, 0].imshow(img[:, :, slice_x], cmap='gray')
    axes[2, 0].set_title('Image - Sagittal')
    
    axes[2, 1].imshow(img[:, :, slice_x], cmap='gray')
    axes[2, 1].imshow(tumor[:, :, slice_x], cmap='hot', alpha=0.5)
    axes[2, 1].set_title('Tumor Mask - Sagittal')
    
    axes[2, 2].imshow(img[:, :, slice_x], cmap='gray')
    axes[2, 2].imshow(ica[:, :, slice_x], cmap='winter', alpha=0.5)
    axes[2, 2].set_title('ICA Mask - Sagittal')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{patient_id}_idx{idx}_planes.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # Also show slices where masks are most prominent
    # Find slices with maximum mask area
    tumor_sum_z = np.sum(tumor, axis=(1, 2))
    tumor_best_z = np.argmax(tumor_sum_z)
    
    ica_sum_z = np.sum(ica, axis=(1, 2))
    ica_best_z = np.argmax(ica_sum_z)
    
    # Show the slices with maximum mask presence
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
    
    axes2[0, 0].imshow(img[tumor_best_z, :, :], cmap='gray')
    axes2[0, 0].set_title(f'Image - Axial (Tumor Max Z={tumor_best_z})')
    
    axes2[0, 1].imshow(img[tumor_best_z, :, :], cmap='gray')
    axes2[0, 1].imshow(tumor[tumor_best_z, :, :], cmap='hot', alpha=0.5)
    axes2[0, 1].set_title(f'Tumor Mask - Axial (Max Z={tumor_best_z})')
    
    axes2[1, 0].imshow(img[ica_best_z, :, :], cmap='gray')
    axes2[1, 0].set_title(f'Image - Axial (ICA Max Z={ica_best_z})')
    
    axes2[1, 1].imshow(img[ica_best_z, :, :], cmap='gray')
    axes2[1, 1].imshow(ica[ica_best_z, :, :], cmap='winter', alpha=0.5)
    axes2[1, 1].set_title(f'ICA Mask - Axial (Max Z={ica_best_z})')
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, f"{patient_id}_idx{idx}_max_slices.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    
    # Save statistics to a text file
    stats_path = os.path.join(output_dir, f"{patient_id}_idx{idx}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Image shape: {img.shape}\n")
        f.write(f"Tumor mask shape: {tumor.shape}\n")
        f.write(f"ICA mask shape: {ica.shape}\n")
        f.write(f"Tumor mask range: [{np.min(tumor)}, {np.max(tumor)}]\n")
        f.write(f"ICA mask range: [{np.min(ica)}, {np.max(ica)}]\n")
        f.write(f"Tumor mask positive voxels: {np.sum(tumor > 0)}\n")
        f.write(f"ICA mask positive voxels: {np.sum(ica > 0)}\n")
    
    print(f"Visualization saved to {output_dir} for patient {patient_id} (index {idx})")
    return img, tumor, ica