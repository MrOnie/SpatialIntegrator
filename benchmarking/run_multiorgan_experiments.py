import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from sklearn.metrics import silhouette_score

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialintegrator.core.dataset import SpatialDataset
from spatialintegrator.models.vision_extractor import ImageEmbedder
from spatialintegrator.models.fusion import ModalityFuser
from spatialintegrator.tl.clustering import cluster_multimodal

SCENARIOS = {
    "Human Breast Cancer": "V1_Breast_Cancer_Block_A_Section_1",
    "Human Lymph Node": "V1_Human_Lymph_Node",
    "Human Brain Cortex": "V1_Human_Brain_Section_1",
    "Adult Mouse Brain": "V1_Adult_Mouse_Brain",
    "Human Heart": "V1_Human_Heart"
}

def load_and_preprocess(dataset_id: str):
    print(f"  -> Fetching dataset from Squidpy/10x Genomics: {dataset_id}...")
    if dataset_id == "V1_Breast_Cancer_Block_A_Section_1":
        adata = sq.datasets.visium_hne_adata()
    else:
        adata = sq.datasets.visium(dataset_id, include_hires_tiff=False)
        
    adata.var_names_make_unique()
    
    # Standard normalization and log-transformation
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    
    adata_rna = adata[:, adata.var.highly_variable].copy()
    if hasattr(adata_rna.X, 'toarray'):
        adata_rna.X = adata_rna.X.toarray()
    sc.pp.scale(adata_rna)
    sc.tl.pca(adata_rna, n_comps=50)
    
    library_id = list(adata.uns['spatial'].keys())[0]
    dataset = SpatialDataset(adata_rna, library_id=library_id)
    return dataset, adata_rna

def evaluate_sss(coords, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= len(coords):
        return -1.0
    return silhouette_score(coords, labels, metric='euclidean')

def run_multiorgan_benchmark():
    os.makedirs("results", exist_ok=True)
    print("="*80)
    print("SPATIALINTEGRATOR: MULTI-ORGAN ARCHITECTURAL BENCHMARKING SUITE")
    print("="*80)
    
    # Initialize vision models once to maximize CPU/GPU runtime efficiency
    print("Initializing Pathology Vision Foundation Models (Phikon & ViT-Base)...")
    embedder_phikon = ImageEmbedder(model_name="phikon")
    embedder_vit = ImageEmbedder(model_name="vit-base")
    fuser = ModalityFuser(n_components=50)
    
    results = []
    
    for organ_name, dataset_id in SCENARIOS.items():
        print(f"\n================================================================================")
        print(f"EVALUATING ORGAN SYSTEM: {organ_name} ({dataset_id})")
        print(f"================================================================================")
        t0 = time.time()
        try:
            dataset, adata_rna = load_and_preprocess(dataset_id)
            n_spots = dataset.adata.n_obs
            coords = dataset.adata.obsm['spatial']
            rna_matrix = dataset.adata.X
            if hasattr(rna_matrix, 'toarray'):
                rna_matrix = rna_matrix.toarray()
                
            print(f"  -> Loaded {n_spots} spots, {adata_rna.n_vars} highly variable genes.")
            
            # 1. Unimodal RNA-Only Baseline
            print("  [1/3] Computing Unimodal RNA Baseline...")
            sc.pp.neighbors(adata_rna, use_rep='X_pca', key_added='rna_neighbors')
            sc.tl.leiden(adata_rna, resolution=0.8, neighbors_key='rna_neighbors', key_added='leiden_rna', flavor='igraph', n_iterations=2, directed=False)
            labels_rna = adata_rna.obs['leiden_rna'].values
            n_dom_rna = len(np.unique(labels_rna))
            sss_rna = evaluate_sss(coords, labels_rna)
            print(f"      * RNA-Only Baseline -> Domains: {n_dom_rna} | Spatial Silhouette Score: {sss_rna:.4f}")
            
            # Extract patches once at 224px
            print("  -> Extracting localized 224x224px histology morphological tiles...")
            patches_224 = dataset.extract_patches(patch_size=224)
            
            # 2. SpatialIntegrator (ViT-Base, 224px, alpha=0.2)
            print("  [2/3] Computing SpatialIntegrator with ViT-Base backbone...")
            embeddings_vit = embedder_vit.extract_embeddings(patches_224, batch_size=32)
            joint_vit = fuser.fit_transform(rna_matrix, embeddings_vit, alpha=0.2)
            cluster_multimodal(dataset.adata, joint_vit, key_added="leiden_vit_224", resolution=0.8)
            labels_vit = dataset.adata.obs['leiden_vit_224'].values
            n_dom_vit = len(np.unique(labels_vit))
            sss_vit = evaluate_sss(coords, labels_vit)
            print(f"      * ViT-Base (224px, alpha=0.2) -> Domains: {n_dom_vit} | Spatial Silhouette Score: {sss_vit:.4f}")

            # 3. SpatialIntegrator (Phikon, 224px, alpha=0.2)
            print("  [3/3] Computing SpatialIntegrator with pathology specialist Phikon backbone...")
            embeddings_phikon = embedder_phikon.extract_embeddings(patches_224, batch_size=32)
            joint_phikon = fuser.fit_transform(rna_matrix, embeddings_phikon, alpha=0.2)
            cluster_multimodal(dataset.adata, joint_phikon, key_added="leiden_phikon_224", resolution=0.8)
            labels_phikon = dataset.adata.obs['leiden_phikon_224'].values
            n_dom_phikon = len(np.unique(labels_phikon))
            sss_phikon = evaluate_sss(coords, labels_phikon)
            print(f"      * Phikon (224px, alpha=0.2) -> Domains: {n_dom_phikon} | Spatial Silhouette Score: {sss_phikon:.4f}")

            elapsed = time.time() - t0
            print(f"  => Scenario completed in {elapsed:.1f}s")
            
            results.append({
                "Organ System": organ_name,
                "Dataset ID": dataset_id,
                "Spots": n_spots,
                "RNA Domains": n_dom_rna,
                "RNA SSS": round(sss_rna, 4),
                "ViT Domains": n_dom_vit,
                "ViT SSS": round(sss_vit, 4),
                "Phikon Domains": n_dom_phikon,
                "Phikon SSS": round(sss_phikon, 4)
            })

        except Exception as e:
            print(f"[ERROR] Failed evaluating {organ_name}: {e}")
            import traceback
            traceback.print_exc()
            
    df_results = pd.DataFrame(results)
    csv_out = "results/multi_organ_benchmark.csv"
    df_results.to_csv(csv_out, index=False)
    print("\n" + "="*80)
    print(f"MULTI-ORGAN BENCHMARK SUMMARY TABLE SAVED TO {csv_out}:")
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Generate Publication-Quality Figure 5: Multi-Organ SSS Comparison
    if len(df_results) > 0:
        print("Generating 300 DPI comparative print visualization...")
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
        
        x = np.arange(len(df_results))
        width = 0.26
        
        rects1 = ax.bar(x - width, df_results['RNA SSS'], width, label='RNA-Only Baseline (Unimodal)', color='#a0aec0', edgecolor='black', linewidth=0.8)
        rects2 = ax.bar(x, df_results['ViT SSS'], width, label='SpatialIntegrator (ViT-Base, 224px)', color='#4299e1', edgecolor='black', linewidth=0.8)
        rects3 = ax.bar(x + width, df_results['Phikon SSS'], width, label='SpatialIntegrator (Phikon Specialist, 224px)', color='#38a169', edgecolor='black', linewidth=0.8)
        
        ax.set_ylabel('Spatial Silhouette Score (Contiguity)', fontsize=13, fontweight='bold')
        ax.set_title('Cross-Organ Multimodal Integration Coherence (10x Visium Reference Suite)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(df_results['Organ System'], fontsize=11, fontweight='bold', rotation=12)
        ax.legend(fontsize=11, frameon=True, facecolor='white', framealpha=0.9, loc='best')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        # Annotate bar values
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                val_str = f"{height:+.3f}" if height != -1.0 else "N/A"
                y_pos = height + 0.006 if height >= 0 else height - 0.015
                ax.annotate(val_str,
                            xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=9.5, fontweight='semibold')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        plt.tight_layout()
        fig_path = "results/fig5_multiorgan_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to {fig_path}")

if __name__ == "__main__":
    run_multiorgan_benchmark()
