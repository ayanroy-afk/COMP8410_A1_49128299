# COMP8410 Assignment 1: Required Models and Datasets

This repository evaluates four pairs of AI models for Amazon retail applications. Large datasets and pretrained model weights are not included in the repository.

For Python dependencies, follow the existing `requirements*.txt` files for each pair.

## Pair 1: Product Image Inpainting

**Dataset:** [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), including product images and metadata.

**Models to download:**
- [BrushNet](https://github.com/TencentARC/BrushNet), using the **random-mask checkpoint**.
- [PowerPaint-v2-1](https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1).
- **realisticVisionV60B1_v51VAE** base model, required by the inpainting setup. The PowerPaint-v2-1 repository includes this model.

## Pair 2: Product Search Relevance

**Dataset:** [Amazon Shopping Queries (ESCI)](https://github.com/amazon-science/esci-data).

**Models to download:**
- [Qwen3.5-4B GGUF](https://huggingface.co/lmstudio-community/Qwen3.5-4B-GGUF), file `Qwen3.5-4B-Q4_K_M.gguf`.
- [Gemma 4 E4B GGUF](https://huggingface.co/nile-academy/gemma-4-e4b-it-gguf), file `gemma-4-E4B_q4_0-it.gguf`.

Local GGUF inference uses [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Pair 3: Personalised Product Recommendations

**Dataset:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), **Musical Instruments** category (reviews and item information).

**Models:** [LightGCL](https://github.com/HKUDS/LightGCL) and [XSimGCL](https://github.com/Coder-Yu/SELFRec). No pretrained checkpoints are required. Both models are trained on the prepared dataset.

## Pair 4: Promotional Image Generation

**Dataset:** [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), using product metadata. The evaluation prompts are provided in this repository.

**Models to download:**
- [Z-Image-Turbo SDNQ uint4](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32).
- [FLUX.2 klein 4B SDNQ 4-bit](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic).

Refer to the relevant notebooks for the expected local file locations.
