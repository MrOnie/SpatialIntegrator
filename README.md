# SpatialIntegrator

**SpatialIntegrator** is a Python framework designed to integrate spatial transcriptomics (e.g., 10x Visium) with high-resolution histological images (H&E). It leverages Vision Foundation Models (such as UNI, Phikon, and ViT) to extract morphological features and fuses them mathematically with gene expression, enabling the discovery of hidden "tissue domains" that are not detectable using RNA alone.

## Features
- **Multimodal Integration:** Jointly models RNA and morphological image features.
- **Foundation Models Support:** Built-in support for state-of-the-art computational pathology models:
  - `vit-base`: Standard Vision Transformer (ImageNet pretrained).
  - `uni`: MahmoodLab's UNI pathology foundation model.
  - `phikon`: Owkin's Phikon pathology foundation model.
- **Interactive Dashboard:** Includes a Streamlit web application for easy visualization.

## Installation

```bash
git clone https://github.com/MrOnie/SpatialIntegrator.git
cd SpatialIntegrator
pip install -e .
```

## Foundation Models & Hugging Face Authentication

Some advanced pathology foundation models are "gated" on Hugging Face. This means you must request access before using them.

### How to use UNI (`MahmoodLab/UNI`)
1. Go to the [MahmoodLab/UNI Hugging Face repository](https://huggingface.co/MahmoodLab/UNI).
2. Accept the terms of use (CC-BY-NC-ND 4.0 license). You may need to use your institutional email.
3. Generate a Hugging Face Access Token in your account settings (`Profile -> Settings -> Access Tokens`).
4. You can provide this token directly in the **SpatialIntegrator Dashboard**, or set it in your terminal before running scripts:
   ```bash
   huggingface-cli login
   ```

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## Running the Benchmark

We provide an automated script to benchmark the multimodal approach against a standard RNA-only pipeline using a public breast cancer dataset from `squidpy`.

```bash
python benchmarking/run_benchmark.py
```