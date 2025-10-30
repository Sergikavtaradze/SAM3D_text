# SAM3DText: Text-Promptable 3D Medical Image Segmentation

SAM3DText extends the 3DSAM-adapter to accept **natural language prompts** alongside traditional point prompts. The project replaces the point prompt encoder with frozen **GPT-2-125m** and **LLaMA-3.2-1b** models for text-guided 3D medical image segmentation.

---

## Overview

* Built on [3DSAM-adapter](https://github.com/med-air/3DSAM-adapter).
* Introduces **text prompt encoders** using GPT-2 and LLaMA-3.2.
* Fuses text and vision features via **cross-attention**.
* Retains SAM-B’s frozen 3D encoder with lightweight adapters.

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
  * `PromptEncoder_Text_Llama`: same overall design a linear projection (2048→256)) using `AutoModel` and `AutoTokenizer` for LLAMA from Hugging Face.
  * `CrossAttentionModel`: fuses text embeddings with 3D features using memory-efficient chunked attention.
* **Added:** optional class embeddings for biasing toward known anatomy classes.

### 2. Training Scripts

#### `train.py`

* Multi-class maskwise training with point prompts.
* Samples positive and negative 3D points per class.
* Uses four prompt encoders; only the deepest layer receives prompt features.

#### `train_GPT.py`

* Replaces point prompts with text prompts.
* Uses `PromptEncoder_Text` (GPT-2) to encode queries like:

  * "Segment the Pituitary Tumor"
  * "Segment the Internal Carotid Artery"
* Fuses text and 3D features via cross-attention.

#### `train_LLAMA.py`

* Same as GPT version, using `PromptEncoder_Text_Llama` and a chosen LLAMA checkpoint.

---

## Results (Should I include a section on results?)

---

## Installation

Project uses CUDA 12.4 PyTorch stack pinned in `requirements.txt`.

```bash
conda create -n sam3dtext python=3.9
conda activate sam3dtext
pip install -r requirements.txt
```

Notes

* PyTorch 2.6 wheels include CUDA 12.4 runtime. An NVIDIA driver with CUDA 12.4 support is required.
* Place `ckpt/sam_vit_b_01ec64.pth` (i.e. SAM checkpoint) before training.

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

Tips

* Keep `128³` for text runs to limit attention memory.
* OR Reduce chunk size in `CrossAttentionModel` if OOM.


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

---
