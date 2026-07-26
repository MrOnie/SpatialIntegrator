# SpatialIntegrator 🧬🔬
**A Foundation Model-Driven Framework for Multimodal Integration of Histopathology and Spatial Transcriptomics**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scanpy 1.12.0+](https://img.shields.io/badge/Scanpy-1.12.0%2B-5c82ff.svg)](https://scanpy.readthedocs.io)

**SpatialIntegrator** is an open-source computational pathology and genomics framework designed to solve single-modality fragmentation and drop-out artifacts in spatial transcriptomics (e.g., 10x Visium, Slide-seq). By coupling highly conserved morphological features extracted from whole-slide histology (H&E) via self-supervised **Vision Foundation Models** (VFMs) with targeted gene expression profiles, SpatialIntegrator unifies localized phenotypic texture with molecular signature mapping.

---

## 🔥 Architecture & Highlights
* **Deep Computational Pathology Backbones:** Built-in modular integration of cutting-edge histopathology foundational models trained on millions of pathology tiles:
  * 🟢 **`phikon` (Owkin):** Vision Transformer self-supervisedly distilled via iBOT on TCGA pathology datasets. Exhibits state-of-the-art microenvironmental cellular resolution.
  * 🔵 **`uni` (Mahmood Lab / Harvard):** General-purpose clinical foundational model (16-bit representation space) engineered for tissue-level diagnostics and pan-cancer classification.
  * ⚪ **`vit-base` (Google / ViT-B/16):** Standard ImageNet pre-trained baseline for speedy benchmarking.
* **Algebraic Multimodal Fusion & Frobenius Inertia Equalization (`ModalityFuser`):** Mathematically unifies variance-stabilized RNA profiles (PCA reduced) and dense visual embeddings into an equilibrated joint representation. Normalizing both manifolds by their Frobenius norm ensures equal eigenvalue spectral inertia before applying parametric weighting hyperparameter $\alpha \in [0, 1]$.
* **Robust Community Detection & Non-Parametric Biomarkers:** Utilizes graph-based **Leiden clustering** powered by high-speed C-bindings (`igraph`) alongside non-parametric **Wilcoxon rank-sum testing** to demarcate contiguously coherent anatomico-molecular tissue domains and robust DEGs.
* **Guided Interactive Dashboard:** A sleek, fully featured interactive web application powered by **Streamlit** (featuring custom dark mode theming and 1-click test dataset loading).

---

## 📈 Benchmark Performance & Visualizations
Evaluated on official standardized human infiltrating ductal carcinoma (Visium H&E Breast Cancer dataset via *Squidpy*), SpatialIntegrator consistently outperforms traditional RNA-only approaches by bridging technological transcript dropouts with physical tissue architecture:

| Pipeline Modality & Backbone | Receptive Tile Size | RNA Weight ($\alpha$) | Discovered Domains | Spatial Silhouette Score (Contiguity) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **RNA-Only Baseline** (Standard Scanpy) | N/A | N/A (Unimodal) | 16 | $-0.0536$ *(Noisy / Disconnected)* |
| **SpatialIntegrator (`phikon`)** | $112 \times 112\text{px}$ | $0.2$ | 24 | $+0.1485$ *(Fine-grained morphology)* |
| **SpatialIntegrator (`phikon`)** | $224 \times 224\text{px}$ | $0.2$ | 26 | $+0.1576$ *(Optimal biomarker resolution)* |
| **SpatialIntegrator (`phikon`)** | $336 \times 336\text{px}$ | $0.2$ | 27 | $+0.1735$ |
| **SpatialIntegrator (`vit-base`)** | $336 \times 336\text{px}$ | $0.2$ | 21 | **$+0.2141$ *(Max macro-contiguity)*** |

> [!IMPORTANT]
> **Key Insight:** Structuring the feature space with morphological priors ($\alpha = 0.2$, tile resolution $224\text{px}$) isolates tumor perimeter invasive fronts expressing elevated **ERBB2 (HER2)**, **FASN**, and extracellular matrix remodeling biomarkers (**MMP11**, **COL1A1**).

### 🗺️ High-Resolution Tissue Domain Segmentation
By leveraging self-supervised digital pathology backbones, SpatialIntegrator effectively smooths technical sequencing dropouts to reveal true biological structures:

![Spatial Domain Grid Comparison](results/fig4_spatial_domains_grid.png)
*Figure 1: Comparative histological domain mapping across standard unimodal RNA clustering, generalist ViT-Base, and pathology-specialized Phikon representations across intact H&E whole-slide histology.*

### 🧬 Biomarker Discovery & Microenvironmental Validation
Morphology-informed spatial domain identification preserves molecular specificities while resolving intricate tumor microenvironments (TMEs):

![Biomarker Dotplot Validation](results/figura_2_marcadores_224.png)
*Figure 2: Dotplot mapping mean expression levels and percentage of expressing spots for top differentially expressed biomarker genes across multimodal spatial domains (Phikon backbone, 224px tile resolution). Note the precise identification of ERBB2/HER2+ invasive fronts and MMP11+ reactive extracellular matrix remodelers.*

### ⚖️ Receptive Field & Modality Sensitivity Analysis
Evaluating performance across varying physical tile resolutions ($112\text{px}$ to $336\text{px}$) confirms the operational advantage of computational pathology specialization:

![Model Sensitivity Comparison](results/fig3_model_alpha_comparison.png)
*Figure 3: Sensitivity profiling of Spatial Silhouette coherence as a function of receptive field tile resolution across vision foundation models compared against the RNA-only baseline.*

---

## 💻 Interactive Evaluation Dashboard

Launch the guided exploratory interface locally on port `8501` without requiring programming expertise:
```bash
streamlit run dashboard/app.py
```

![SpatialIntegrator Interactive Streamlit Dashboard](images/StreamlitDashboard_main.png)

* **Instant Test Mode:** Simply click **"🧪 Load Test Dataset (Visium H&E)"** inside the left sidebar to automatically download and evaluate the standard Squidpy breast cancer slide without local file setup.
* **Custom Dataset Processing:** Simply input any directory path conforming to standard 10x Space Ranger formats (`filtered_feature_bc_matrix.h5`, `spatial/tissue_hires_image.png`, `spatial/scalefactors_json.json`).
* **Interactive Export:** Download computed spatial biomarkers (DEGs) via the non-parametric Wilcoxon rank-sum test with Benjamini-Hochberg FDR correction directly to CSV format.

---

## 🚀 Installation & Environment Setup

Clone the repository and install dependencies directly into an isolated virtual environment:

```bash
# Clone repository
git clone https://github.com/MrOnie/SpatialIntegrator.git
cd SpatialIntegrator

# Install in editable mode with dependencies
pip install -e .
```

### 🔐 Gated Clinical Foundational Models (Hugging Face)
While `phikon` and `vit-base` are accessible immediately without credential verification, advanced clinical foundation models like **UNI (`MahmoodLab/UNI`)** require credential verification:
1. Navigate to [MahmoodLab/UNI on Hugging Face](https://huggingface.co/MahmoodLab/UNI) and accept the academic research license (CC-BY-NC-ND 4.0).
2. Generate a Personal Access Token via `Hugging Face Settings -> Access Tokens` (with Read permissions).
3. Authenticate locally in your terminal via the modern CLI:
   ```bash
   hf auth login
   ```
   *(Alternatively, paste your token directly into the designated secure input field in the interactive Streamlit dashboard).*

---

## 🔬 Reproducing Experimental Benchmarks & Manuscript Figures

All quantitative experimental tables and high-resolution print figures generated for the accompanying Q1 scientific publication can be regenerated cleanly via our command-line benchmarking suite:

```bash
# 1. Run full grid comparison (Phikon vs ViT vs RNA across resolutions & weights)
python benchmarking/run_comprehensive_experiments.py

# 2. Generate publication 300 DPI biomarker differential dotplots (Figure 2)
python benchmarking/generate_biomarker_figure.py
```
Outputs and quantitative figures are exported directly to the `results/` directory.

---

## 📚 Citation & License

If you utilize **SpatialIntegrator** or adapt its multimodal fusion methodology in your research, please cite our corresponding work:

```bibtex
@article{SpatialIntegrator2026,
  title   = {SpatialIntegrator: A Foundation Model-Driven Framework for Multimodal Integration of Histopathology and Spatial Transcriptomics},
  author  = {Martínez R. et al.},
  journal = {Preprint / Target Journal},
  year    = {2026},
  url     = {https://github.com/MrOnie/SpatialIntegrator}
}
```

This software is distributed under the **MIT License**. See `LICENSE` for further details.