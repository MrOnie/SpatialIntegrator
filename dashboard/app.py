import streamlit as st
import os
import sys

# Add parent dir to path to import spatialintegrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialintegrator.core.dataset import SpatialDataset
from spatialintegrator.models.vision_extractor import ImageEmbedder
from spatialintegrator.models.fusion import ModalityFuser
from spatialintegrator.tl.clustering import cluster_multimodal
from spatialintegrator.pl.visualization import plot_spatial_domains, plot_joint_umap

import matplotlib.pyplot as plt

st.set_page_config(page_title="SpatialIntegrator Dashboard", layout="wide")

st.title("SpatialIntegrator 🧬🔬")
st.markdown("### Multimodal Spatial Transcriptomics & Pathology Integration")

# Sidebar
st.sidebar.header("Data Loading")
dataset_path = st.sidebar.text_input("Path to Visium dataset directory:", value="")

if st.sidebar.button("Load Dataset"):
    if os.path.exists(dataset_path):
        with st.spinner("Loading dataset and preprocessing RNA..."):
            try:
                st.session_state['dataset'] = SpatialDataset.from_visium(dataset_path)
                st.session_state['dataset'].preprocess_rna()
                st.success("Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    else:
        st.error("Directory does not exist. Please provide a valid path.")

if 'dataset' in st.session_state:
    dataset = st.session_state['dataset']
    st.write(f"**Loaded AnnData:** {dataset.adata.n_obs} spots × {dataset.adata.n_vars} genes")
    
    st.sidebar.header("Pipeline Parameters")
    
    # Model Selection
    model_choice = st.sidebar.selectbox(
        "Vision Foundation Model",
        options=['vit-base', 'phikon', 'uni'],
        index=0,
        help="'uni' and 'phikon' are specialized pathology models."
    )
    
    # HF Token Input (only needed for gated models like UNI)
    hf_token = ""
    if model_choice in ['uni']:
        hf_token = st.sidebar.text_input("Hugging Face Token (Required for UNI):", type="password", 
                                         help="UNI is a gated model. Get your token at https://huggingface.co/settings/tokens")
    
    patch_size = st.sidebar.slider("Image Patch Size", 112, 512, 224, step=16)
    alpha = st.sidebar.slider("RNA Weight (Alpha)", 0.0, 1.0, 0.5, step=0.1)
    resolution = st.sidebar.slider("Clustering Resolution", 0.1, 2.0, 1.0, step=0.1)
    
    if st.button("Run Multimodal Integration Pipeline"):
        try:
            with st.spinner("1. Extracting Image Patches..."):
                patches = dataset.extract_patches(patch_size=patch_size)
                
            with st.spinner(f"2. Extracting Deep Morphological Features ({model_choice})..."):
                embedder = ImageEmbedder(model_name=model_choice, hf_token=hf_token if hf_token else None)
                img_embeddings = embedder.extract_embeddings(patches)
                
            with st.spinner("3. Fusing Modalities..."):
                fuser = ModalityFuser()
                # Need dense matrix for RNA
                rna_matrix = dataset.adata.X.toarray() if hasattr(dataset.adata.X, 'toarray') else dataset.adata.X
                joint_rep = fuser.fit_transform(rna_matrix, img_embeddings, alpha=alpha)
                
            with st.spinner("4. Clustering..."):
                adata_res = cluster_multimodal(dataset.adata, joint_rep, resolution=resolution)
                st.session_state['adata_res'] = adata_res
                st.success("Pipeline completed!")
        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {e}")
            
if 'adata_res' in st.session_state:
    st.header("Results Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spatial Domains (H&E Overlay)")
        fig_spatial, _ = plot_spatial_domains(st.session_state['adata_res'])
        st.pyplot(fig_spatial)
        
    with col2:
        st.subheader("Joint UMAP Latent Space")
        fig_umap = plot_joint_umap(st.session_state['adata_res'])
        st.pyplot(fig_umap)
