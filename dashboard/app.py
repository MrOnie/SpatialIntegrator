import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import scanpy as sc
import squidpy as sq

# Add parent dir to path to import spatialintegrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialintegrator.core.dataset import SpatialDataset
from spatialintegrator.models.vision_extractor import ImageEmbedder
from spatialintegrator.models.fusion import ModalityFuser
from spatialintegrator.tl.clustering import cluster_multimodal
from spatialintegrator.pl.visualization import plot_spatial_domains, plot_joint_umap

st.set_page_config(
    page_title="SpatialIntegrator Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# RESOURCE CACHING (Performance Optimizations & Dataset Vault)
# -----------------------------------------------------------------------------
BENCHMARK_SCENARIOS = {
    "Human Breast Cancer (Invasive Ductal Carcinoma)": {
        "id": "V1_Breast_Cancer_Block_A_Section_1",
        "desc": "**Oncology & Tumor Microenvironment:** Differentiates invasive ductal carcinomatosis lobulate fronts from desmoplastic reactive stroma and extracellular connective matrix (biomarkers: *ERBB2*, *MMP11*, *FASN*)."
    },
    "Human Lymph Node (Immunology & Architecture)": {
        "id": "V1_Human_Lymph_Node",
        "desc": "**Immunology & Secondary Lymphoid Organs:** Resolves germinal centers, T-cell rich paracortex zones, and lymphoid follicles. Ideal benchmark for validating naturally dispersed, multifocal physiological tissue domains without artificial smoothing."
    },
    "Human Brain Cortex (Dorsolateral Prefrontal Cortex)": {
        "id": "V1_Human_Brain_Section_1",
        "desc": "**Neuropathology & Neuroscience:** Maps fine cortical laminar organization (Layers L1 through L6) and subcortical deep white matter neuronal tracts."
    },
    "Adult Mouse Brain (Whole-Brain Sagittal Section)": {
        "id": "V1_Adult_Mouse_Brain",
        "desc": "**Complex Multi-Regional Neuroanatomy:** Complete macro-architectural mapping of the hippocampus, thalamus, cerebellum, and cerebral ventricles. The gold standard analytical benchmark in computational spatial omics literature."
    },
    "Human Heart (Cardiomyocyte & Fibrotic Myocardium)": {
        "id": "V1_Human_Heart",
        "desc": "**Cardiovascular Precision Medicine:** Evaluates cardiomyocyte structural myofibril orientation, vascular architecture, and interstitial fibrosis niches."
    }
}

@st.cache_resource(show_spinner=False)
def load_benchmark_dataset(dataset_id: str):
    """Downloads and processes standardized benchmark Visium datasets from 10x Genomics via Squidpy."""
    if dataset_id == "V1_Breast_Cancer_Block_A_Section_1":
        adata = sq.datasets.visium_hne_adata()
    else:
        adata = sq.datasets.visium(dataset_id, include_hires_tiff=False)
        
    adata.var_names_make_unique()
    
    # Standard Scanpy normalization and log-transformation
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    
    adata_rna = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_rna)
    sc.tl.pca(adata_rna)
    
    library_id = list(adata.uns['spatial'].keys())[0]
    dataset = SpatialDataset(adata_rna, library_id=library_id)
    return dataset, adata

@st.cache_resource(show_spinner=False)
def get_vision_embedder(model_name: str, hf_token: str = None):
    """Loads and caches the vision foundation model embedder."""
    return ImageEmbedder(model_name=model_name, hf_token=hf_token if hf_token else None)

# -----------------------------------------------------------------------------
# HEADER & STEP-BY-STEP GUIDED WORKFLOW
# -----------------------------------------------------------------------------
st.title("🔬 SpatialIntegrator")
st.markdown("#### Foundation Model-Driven Multimodal Integration Platform for Spatial Transcriptomics & Histopathology")

with st.expander("👉 Step-by-Step Guide: How to run benchmark evaluation examples (Click to toggle)", expanded=True):
    st.markdown("""
    **Welcome to the interactive evaluation dashboard!** You can instantly explore SpatialIntegrator across diverse tissue organ systems without needing any local dataset configuration:
    
    1. **Select Biological Scenario:** On the left sidebar under *1. 🗄️ Data Source*, choose from five curated 10x Visium clinical and physiological reference datasets (Breast Cancer, Lymph Node, Brain Cortex, Whole Mouse Brain, or Heart).
    2. **Instant Data Loading:** Click the primary button **"🧪 Load Selected Benchmark Dataset"**. The platform automatically fetches the raw H&E high-resolution tissue slice and gene count arrays, performs highly variable gene isolation ($k=3000$), and equalizes spectral variance via our Frobenius Norm fuser.
    3. **Select Vision Foundation Model:** Under *2. Pipeline Parameters*, pick your pathology foundation model backbone. We strongly recommend **`phikon`** (distilled on TCGA whole-slide histology images via iBOT self-supervision) for complex oncology and cellular histology.
    4. **Calibrate Multimodal Balance (α):** Adjust the **"RNA Modality Weight (Alpha)"** slider to tune modality synergy:
       * **α = 0.2** *(Recommended Default)*: Assigns 80% weight to H&E morphological architecture (ideal for sharpening contiguous anatomical borders and defining distinct tissue niches).
       * **α = 0.5**: Symmetric balanced concatenation.
       * **α = 0.8**: Transcriptome-dominated representation.
    5. **Execute Multimodal Pipeline:** Click **"⚡ Run Multimodal Pipeline"**. In seconds, the application computes joint representations, evaluates neighborhood graphs, and performs Leiden clustering.
    6. **Validate & Export Biomarkers:** Navigate across the bottom interactive tabs to inspect histology domain overlays, analyze Joint UMAP embeddings, and execute non-parametric **Wilcoxon Rank-Sum tests** to discover and download table of differentially expressed genes (DEGs) to CSV.
    """)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. 🗄️ Data Source")
    
    # Multi-scenario Benchmark Selector
    st.subheader("🧪 Official Benchmark Datasets")
    selected_scenario_name = st.selectbox(
        "Select Biological Test Scenario:",
        list(BENCHMARK_SCENARIOS.keys()),
        index=0,
        help="Choose a pre-configured 10x Genomics Visium benchmark dataset."
    )
    
    selected_info = BENCHMARK_SCENARIOS[selected_scenario_name]
    st.markdown(f"<div style='font-size: 0.85em; background-color: rgba(255, 255, 255, 0.05); padding: 8px; border-radius: 5px; margin-bottom: 10px;'>{selected_info['desc']}</div>", unsafe_allow_html=True)
    
    if st.button("🧪 Load Selected Benchmark Dataset", type="primary", use_container_width=True):
        with st.spinner(f"Downloading & processing {selected_scenario_name}..."):
            try:
                dataset, full_adata = load_benchmark_dataset(selected_info["id"])
                st.session_state['dataset'] = dataset
                st.session_state['full_adata'] = full_adata
                st.session_state['dataset_source'] = selected_scenario_name
                st.success("Benchmark dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading benchmark dataset: {e}")

    st.markdown("---")
    st.subheader("📂 Custom Local Visium Data")
    st.caption("Or specify a local directory (10x Genomics Visium standard folder format):")
    dataset_path = st.text_input("Visium directory path", placeholder="C:/path/to/visium_output", label_visibility="collapsed")
    if st.button("📂 Load Local Directory", use_container_width=True):
        if os.path.exists(dataset_path):
            with st.spinner("Processing local count matrix and spatial coordinates..."):
                try:
                    ds = SpatialDataset.from_visium(dataset_path)
                    ds.preprocess_rna()
                    st.session_state['dataset'] = ds
                    st.session_state['full_adata'] = ds.adata
                    st.session_state['dataset_source'] = f"Local: {os.path.basename(dataset_path)}"
                    st.success("Local dataset imported successfully!")
                except Exception as e:
                    st.error(f"Error importing local dataset: {e}")
        else:
            st.warning("Please provide a valid local directory path.")
    st.markdown("---")
            
    st.header("2. ⚙️ Pipeline Parameters")
    
    model_choice = st.selectbox(
        "Vision Foundation Model",
        options=['phikon', 'vit-base', 'uni'],
        index=0,
        help="Phikon and UNI are domain-specific models pre-trained on millions of histopathology slide tiles. ViT-Base serves as an ImageNet baseline."
    )
    
    hf_token = ""
    if model_choice == 'uni':
        hf_token = st.text_input("Hugging Face Access Token (Required for UNI)", type="password", help="UNI is a gated clinical foundation model requiring institutional approval via Hugging Face.")
        
    patch_size = st.slider("H&E Patch Resolution (px)", min_value=112, max_value=448, value=224, step=16, help="224px represents the optimal physical receptive field balancing cellular detail and surrounding extracellular stroma context.")
    alpha = st.slider("RNA Modality Weight (Alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Values approaching 0 prioritize histological texture coherence; values approaching 1 prioritize gene expression divergence.")
    resolution = st.slider("Leiden Clustering Resolution", min_value=0.2, max_value=2.0, value=1.0, step=0.2, help="Higher resolutions identify finer-grained cellular microenvironmental sub-niches.")
    
    st.markdown("---")
    run_disabled = 'dataset' not in st.session_state
    run_btn = st.button("⚡ Run Multimodal Pipeline", type="primary", disabled=run_disabled)

# -----------------------------------------------------------------------------
# MAIN VIEW (Active Data & Execution State)
# -----------------------------------------------------------------------------
if 'dataset' in st.session_state:
    dataset = st.session_state['dataset']
    source_name = st.session_state.get('dataset_source', 'Visium Dataset')
    
    with st.container(border=True):
        st.subheader(f"✅ Active Dataset: **{source_name}**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sequenced Spots", f"{dataset.adata.n_obs:,}")
        col2.metric("Analyzed Features (Genes)", f"{dataset.adata.n_vars:,}")
        col3.metric("Selected Backbone", model_choice.upper())
        col4.metric("Multimodal Ratio", f"{(1-alpha)*100:.0f}% Vision / {alpha*100:.0f}% RNA")

    # Pipeline execution handler
    if run_btn:
        with st.container(border=True):
            st.write("### 🚀 Executing Multimodal Integration...")
            progress_bar = st.progress(0, text="Initializing workflow...")
            
            try:
                progress_bar.progress(25, text=f"Extracting histological patches ({patch_size}x{patch_size} px)...")
                patches = dataset.extract_patches(patch_size=patch_size)
                
                progress_bar.progress(50, text=f"Extracting deep visual representations via {model_choice.upper()}...")
                embedder = get_vision_embedder(model_name=model_choice, hf_token=hf_token if hf_token else None)
                img_embeddings = embedder.extract_embeddings(patches, batch_size=32)
                
                progress_bar.progress(75, text=f"Fusing latent modalities (Alpha = {alpha})...")
                fuser = ModalityFuser(n_components=50)
                rna_matrix = dataset.adata.X.toarray() if hasattr(dataset.adata.X, 'toarray') else dataset.adata.X
                joint_rep = fuser.fit_transform(rna_matrix, img_embeddings, alpha=alpha)
                
                progress_bar.progress(90, text=f"Computing igraph connectivity and Leiden community clustering (Resolution = {resolution})...")
                key_added = f"cluster_{model_choice}_{patch_size}_a{int(alpha*10)}"
                adata_res = cluster_multimodal(dataset.adata, joint_rep, key_added=key_added, resolution=resolution)
                
                # Compute spatial silhouette index for contiguity evaluation
                sil_score = silhouette_score(adata_res.obsm['spatial'], adata_res.obs[key_added])
                
                # Calculate differentially expressed genes (DEGs) per domain via non-parametric Wilcoxon rank-sum test
                rank_key = "biomarkers_deg"
                sc.tl.rank_genes_groups(adata_res, key_added, method="wilcoxon", key_added=rank_key, n_genes=10)
                
                # Format biomarkers as a clean Pandas DataFrame
                result_deg = adata_res.uns[rank_key]
                groups = result_deg['names'].dtype.names
                deg_dict = {}
                for grp in groups:
                    deg_dict[f"Domain {grp} - Gene"] = result_deg['names'][grp]
                    deg_dict[f"Domain {grp} - pval_adj"] = [f"{p:.2e}" for p in result_deg['pvals_adj'][grp]]
                df_deg = pd.DataFrame(deg_dict)

                st.session_state['adata_res'] = adata_res
                st.session_state['cluster_key'] = key_added
                st.session_state['sil_score'] = sil_score
                st.session_state['df_deg'] = df_deg
                st.session_state['last_model'] = model_choice
                st.session_state['last_patch'] = patch_size
                st.session_state['last_alpha'] = alpha
                
                progress_bar.progress(100, text="Integration completed successfully!")
                st.success("✅ **Multimodal Pipeline Execution Finished.** Explore downstream visualizations and gene discoveries below.")
            except Exception as e:
                st.error(f"❌ An exception occurred during pipeline execution: {e}")

# -----------------------------------------------------------------------------
# DOWNSTREAM VISUALIZATION & BIOMARKER DISCOVERY TABS
# -----------------------------------------------------------------------------
if 'adata_res' in st.session_state:
    st.markdown("---")
    st.header("3. 📊 Multimodal Results Exploration")
    
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Spatial Domain Map (H&E Overlay)", 
        "🌌 Joint UMAP Latent Space",
        "🧬 Biomarker Discovery (DEGs)"
    ])
    
    with tab1:
        with st.container(border=True):
            col_info, col_chart = st.columns([1, 3])
            with col_info:
                st.metric(
                    label="Spatial Contiguity (Silhouette Score)", 
                    value=f"{st.session_state['sil_score']:.4f}",
                    delta="Coherent Anatomy" if st.session_state['sil_score'] > 0 else "Noisy/Fragmented"
                )
                st.caption(f"**Discovered Microenvironments:** {len(st.session_state['adata_res'].obs[st.session_state['cluster_key']].unique())} domains")
                st.markdown("""
                **Domain Map Interpretation:**
                Each color highlights an identified tissue niche, discovered by uniting localized histological cellular texture with underlying transcriptional profiling. Note how morphological awareness smooths technical RNA dropout artifacts.
                """)
            with col_chart:
                fig_spatial, _ = plot_spatial_domains(
                    st.session_state['adata_res'], 
                    color=st.session_state['cluster_key'],
                    title=f"Multimodal Tissue Domains ({st.session_state['last_model'].upper()}, {st.session_state['last_patch']}px, Alpha={st.session_state['last_alpha']})"
                )
                st.pyplot(fig_spatial)
                
    with tab2:
        with st.container(border=True):
            st.markdown("#### Manifold Embedding in Joint Latent Representation (UMAP)")
            st.caption("Spots converging in proximity share highly aligned histopathological structures and genomic transcript expression profiles.")
            fig_umap = plot_joint_umap(st.session_state['adata_res'], color=st.session_state['cluster_key'])
            st.pyplot(fig_umap)
            
    with tab3:
        with st.container(border=True):
            st.markdown("#### Top Differentially Expressed Biomarkers per Domain (DEGs)")
            st.caption("Computed using the non-parametric Wilcoxon rank-sum test (Mann-Whitney U) with Benjamini-Hochberg false discovery rate (FDR) correction against all remaining spatial regions, accounting for sparse zero-inflated omics distributions.")
            
            df_deg = st.session_state['df_deg']
            st.dataframe(df_deg)
            
            # Export button
            csv_data = df_deg.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export DEG Biomarker Table (CSV)",
                data=csv_data,
                file_name=f"biomarkers_deg_{st.session_state['last_model']}.csv",
                mime="text/csv",
                type="primary"
            )
else:
    if 'dataset' not in st.session_state:
        st.info("👈 **To begin evaluation, click '🧪 Load Test Dataset (Visium H&E)' in the left controls bar.**")
