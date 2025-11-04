import os
import pickle
import glob

def load_endonasal_data(directory):
    """Load dataset files (no subfolders): for each prefix, pair image, tumor, and carotid mask."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    # Find all files
    image_files = glob.glob(os.path.join(directory, "*_T1.nii.gz"))
    tumor_mask_files = glob.glob(os.path.join(directory, "*_tumor_f.nii.gz"))
    carotid_mask_files = glob.glob(os.path.join(directory, "*_carotids_f.nii.gz"))

    # Build prefix to file maps (no lambda or dict comprehesion)
    image_map = {}
    for f in image_files:
        prefix = os.path.basename(f).replace("_T1.nii.gz", "")
        image_map[prefix] = f
    tumor_map = {}
    for f in tumor_mask_files:
        prefix = os.path.basename(f).replace("_tumor_f.nii.gz", "")
        tumor_map[prefix] = f
    carotid_map = {}
    for f in carotid_mask_files:
        prefix = os.path.basename(f).replace("_carotids_f.nii.gz", "")
        carotid_map[prefix] = f

    # Intersection of all keys (prefixes)
    common_keys = set(image_map.keys()) & set(tumor_map.keys()) & set(carotid_map.keys())
    if not common_keys:
        print("Warning: No matching triplets of image/tumor/carotid mask found in directory root.")
        return [], [], []
    # Collect matching triplets
    images = []
    tumors = []
    carotids = []
    for k in sorted(common_keys):
        images.append(image_map[k])
        tumors.append(tumor_map[k])
        carotids.append(carotid_map[k])
    print(f"Found {len(images)} matched image-tumor-carotid triplets by prefix in {directory}")
    return images, tumors, carotids

if __name__ == "__main__":
    # ---- Hardcoded settings ----
    DATA_DIR = "/Users/sirbucks/projects/3DSAM-adapter-copy/Sample_Dataset"
    OUTPUT_PATH = "/Users/sirbucks/projects/3DSAM-adapter-copy/datafile/split.pkl"
    # ---- Main logic ----
    images, tumor_masks, carotid_masks = load_endonasal_data(DATA_DIR)
    if not images or not tumor_masks or not carotid_masks:
        raise ValueError("No fully paired image/tumor mask/carotid mask triplets found in the directory root")
    rel_images = [os.path.relpath(img, DATA_DIR) for img in images]
    rel_tumor_masks = [os.path.relpath(msk, DATA_DIR) for msk in tumor_masks]
    rel_carotid_masks = [os.path.relpath(msk, DATA_DIR) for msk in carotid_masks]
    splits = {
        0: {
            'test': {}
        }
    }
    for idx, (img, tumor, carotid) in enumerate(zip(rel_images, rel_tumor_masks, rel_carotid_masks)):
        splits[0]['test'][idx] = [img, tumor, carotid]
    print(f"\nAll {len(rel_images)} cases saved as the 'test' set.")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(splits, f)
    print(f"\nTest-only split saved to: {OUTPUT_PATH}")
    print("\nVerifying test set:")
    for idx in list(splits[0]['test'].keys())[:5]:
        print(f"Case {idx}:")
        print(f"  Image:   {splits[0]['test'][idx][0]}")
        print(f"  Tumor:   {splits[0]['test'][idx][1]}")
        print(f"  Carotid: {splits[0]['test'][idx][2]}")