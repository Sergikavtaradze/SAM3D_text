# SAM3DText: Text-Promptable 3D Medical Image Segmentation

SAM3DText extends the 3DSAM-adapter to accept **natural language prompts** alongside traditional point prompts. The project replaces the point prompt encoder with frozen **GPT-2** and **LLaMA-3.2** models for text-guided 3D medical image segmentation.

---

## Overview

* Built on [3DSAM-adapter](https://github.com/med-air/3DSAM-adapter)
* Introduces **text prompt encoders** using GPT-2 and LLaMA-3.2
* Fuses text and vision features via **cross-attention**
* Retains SAM-B’s frozen 3D encoder with lightweight adapters

### Architecture Changes

| Component                | Original              | SAM3DText Modification                                                                 |
| ------------------------ | --------------------- | -------------------------------------------------------------------------------------- |
| `prompt_encoder.py`      | Point-only encoder    | Added `PromptEncoder_Text` and `PromptEncoder_Text_Llama` with GPT-2 and LLaMA support |
| `train_mobarak.py`       | Point prompt training | Multi-class maskwise point training                                                    |
| `train_mobarak_GPT.py`   | N/A                   | Text prompt training using GPT-2 encoder                                               |
| `train_mobarak_LLAMA.py` | N/A                   | Text prompt training using LLaMA-3.2 encoder                                           |

---

## Key Changes

### 1. `prompt_encoder.py`

* **New modules:**

  * `PromptEncoder_Text`: uses `GPT2Model` (frozen) + linear projection (768→256) + cross-attention with 3D features.
  * `PromptEncoder_Text_Llama`: same design with `AutoModel` and `AutoTokenizer` for LLaMA.
  * `CrossAttentionModel`: fuses text embeddings with 3D features using memory-efficient chunked attention.
* **Added:** optional class embeddings for biasing toward known anatomy classes.

### 2. Training Scripts

#### `train_mobarak.py`

* Multi-class maskwise training with point prompts.
* Samples positive and negative 3D points per class.
* Uses four prompt encoders; only the deepest layer receives prompt features.

#### `train_mobarak_GPT.py`

* Replaces point prompts with text prompts.
* Uses `PromptEncoder_Text` (GPT-2) to encode queries like:

  * "Segment the Pituitary Tumor"
  * "Segment the Internal Carotid Artery"
* Fuses text and 3D features via cross-attention.

#### `train_mobarak_LLAMA.py`

* Same as GPT version, using `PromptEncoder_Text_Llama` and a chosen LLaMA checkpoint.

---

## Results (Pituitary MRI)

| Method           | Patch | Tumor Dice | ICA Dice |
| ---------------- | ----- | ---------- | -------- |
| Point prompt     | 128³  | 71.3       | 62.1     |
| Text (GPT-2)     | 128³  | **76.2**   | 59.9     |
| Text (LLaMA-3.2) | 128³  | 75.0       | 58.4     |

* Text prompts improve **tumor** segmentation.
* Point prompts remain stronger for **ICA**.

---

## Installation

```bash
conda create -n sam3dtext python=3.9.16
conda activate sam3dtext

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/deepmind/surface-distance.git
pip install -r requirements.txt
pip install transformers==4.44.0
```

Place `sam_vit_b_01ec64.pth` under `ckpt/`.

---

## Training

**Point Prompt**

```bash
python train_mobarak.py --data maskwise_endonasal --snapshot_path /exp --data_prefix /data --rand_crop_size 128 128 128
```

**Text Prompt (GPT-2)**

```bash
python train_mobarak_GPT.py --data maskwise_endonasal --snapshot_path /exp --data_prefix /data --rand_crop_size 128 128 128
```

**Text Prompt (LLaMA-3.2)**

```bash
python train_mobarak_LLAMA.py --data maskwise_endonasal --snapshot_path /exp --data_prefix /data --rand_crop_size 128 128 128 --llama_model meta-llama/Llama-3.2-1B
```

---

## Evaluation

**Point Prompt:**

```bash
python test_mobarak.py --data maskwise_endonasal --snapshot_path /checkpoints --data_prefix /data --num_prompts 1
```

**Text Prompt:**

```bash
python test_GPT_maskwise.py --data maskwise_endonasal --snapshot_path /checkpoints --data_prefix /data
```

---

## Citation

```bibtex
@article{Gong20233DSAMadapterHA,
  title={3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation},
  author={Gong, Shizhan and Zhong, Yuan and Ma, Wenao and Li, Jinpeng and Wang, Zhao and Zhang, Jingyang and Heng, Pheng-Ann and Dou, Qi},
  journal={arXiv preprint arXiv:2306.13465},
  year={2023}
}

@misc{Kavtaradze2025SAM3DText,
  title={SAM3DText: Text-Prompted Interactive Model for MRI Pituitary Tumor Segmentation},
  author={Kavtaradze, Sergi and Clarkson, Matt and Hoque, Mobarak I},
  year={2025},
  howpublished={Project report}
}
```

---

## Acknowledgements

Based on Segment-Anything and 3DSAM-adapter. Preprocessing with MONAI. Text integration inspired by recent multi-modal foundation models.
