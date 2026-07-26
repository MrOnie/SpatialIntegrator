import os
import sys
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

# Add local package to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatialintegrator.core.dataset import SpatialDataset
from spatialintegrator.models.vision_extractor import ImageEmbedder
from spatialintegrator.models.fusion import ModalityFuser
from spatialintegrator.tl.clustering import cluster_multimodal

def main():
    os.makedirs('results', exist_ok=True)
    print("="*70)
    print("SPATIALINTEGRATOR: COMPREHENSIVE MULTIMODAL BENCHMARKING SUITE")
    print("="*70)
    
    print("\n1. Loading Visium H&E dataset from Squidpy (Human Breast Cancer)...")
    adata = sq.datasets.visium_hne_adata()
    
    print("2. Preprocessing RNA modality...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    
    adata_rna = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_rna)
    sc.tl.pca(adata_rna)
    
    # Baseline: Classical RNA Clustering
    print("\n--- BASELINE: CLASSICAL RNA-ONLY CLUSTERING ---")
    sc.pp.neighbors(adata_rna, use_rep='X_pca', key_added='rna_neighbors')
    sc.tl.leiden(adata_rna, resolution=1.0, neighbors_key='rna_neighbors', key_added='rna_leiden', flavor='igraph', n_iterations=2, directed=False)
    
    sil_rna = silhouette_score(adata_rna.obsm['spatial'], adata_rna.obs['rna_leiden'])
    n_clusters_rna = len(adata_rna.obs['rna_leiden'].unique())
    print(f"[Baseline RNA-Only] Spatial Silhouette Score: {sil_rna:.4f} | Clusters: {n_clusters_rna}")
    
    adata.obs['rna_leiden'] = adata_rna.obs['rna_leiden']
    
    # Setup for Multimodal Experiments
    library_id = list(adata.uns['spatial'].keys())[0]
    dataset = SpatialDataset(adata_rna, library_id=library_id)
    fuser = ModalityFuser(n_components=50)
    
    rna_matrix = dataset.adata.X
    if hasattr(rna_matrix, 'toarray'):
        rna_matrix = rna_matrix.toarray()
        
    results = [
        {
            'Model': 'RNA-Only',
            'Patch_Size': 'N/A',
            'Alpha': 1.0,
            'Spatial_Silhouette': sil_rna,
            'Num_Clusters': n_clusters_rna,
            'Key_Obs': 'rna_leiden'
        }
    ]
    
    models_to_test = ['vit-base', 'phikon']
    patch_sizes = [112, 224, 336]
    alphas = [0.2, 0.5, 0.8]
    
    best_vit_key = None
    best_vit_score = -1.0
    best_phikon_key = None
    best_phikon_score = -1.0
    
    for model_name in models_to_test:
        print(f"\n==================================================")
        print(f"INITIALIZING VISION FOUNDATION MODEL: {model_name.upper()}")
        print(f"==================================================")
        try:
            embedder = ImageEmbedder(model_name=model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
            
        for patch_size in patch_sizes:
            print(f"\n[Model: {model_name} | Patch Size: {patch_size}px] Extracting image patches...")
            patches = dataset.extract_patches(patch_size=patch_size)
            
            print(f"[Model: {model_name} | Patch Size: {patch_size}px] Computing morphological embeddings...")
            img_embeddings = embedder.extract_embeddings(patches, batch_size=32)
            
            for alpha in alphas:
                key_added = f"multi_{model_name}_{patch_size}_a{int(alpha*10)}"
                print(f" -> Fusing space (alpha={alpha}). Clustering in joint space...")
                joint_rep = fuser.fit_transform(rna_matrix, img_embeddings, alpha=alpha)
                
                adata_res = cluster_multimodal(dataset.adata, joint_rep, key_added=key_added, resolution=1.0)
                
                sil_score = silhouette_score(adata_res.obsm['spatial'], adata_res.obs[key_added])
                n_clusters = len(adata_res.obs[key_added].unique())
                
                print(f"    * {key_added}: Silhouette Score = {sil_score:.4f} (Clusters: {n_clusters})")
                
                adata.obs[key_added] = adata_res.obs[key_added].copy()
                results.append({
                    'Model': model_name,
                    'Patch_Size': str(patch_size),
                    'Alpha': alpha,
                    'Spatial_Silhouette': sil_score,
                    'Num_Clusters': n_clusters,
                    'Key_Obs': key_added
                })
                
                if model_name == 'vit-base' and sil_score > best_vit_score:
                    best_vit_score = sil_score
                    best_vit_key = key_added
                elif model_name == 'phikon' and sil_score > best_phikon_score:
                    best_phikon_score = sil_score
                    best_phikon_key = key_added

    # Save results table
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='Spatial_Silhouette', ascending=False)
    df_res.to_csv('results/experiment_summary_table.csv', index=False)
    print("\n" + "="*70)
    print("SUMMARY OF EXPERIMENTAL RESULTS:")
    print(df_res[['Model', 'Patch_Size', 'Alpha', 'Spatial_Silhouette', 'Num_Clusters']].to_string(index=False))
    print("="*70)
    
    # Export LaTeX Table Snippet
    with open('results/latex_table.tex', 'w') as f:
        f.write("\\begin{table}[h!]\n\\centering\n\\begin{tabular}{l c c c c}\n\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Patch Size (px)} & \\textbf{Alpha (RNA Weight)} & \\textbf{Spatial Silhouette} & \\textbf{Clusters} \\\\\\hline\n")
        for _, row in df_res.iterrows():
            f.write(f"{row['Model']} & {row['Patch_Size']} & {row['Alpha']:.1f} & {row['Spatial_Silhouette']:.4f} & {row['Num_Clusters']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Quantitative comparison of spatial domain coherence across foundation models, morphological field of views, and modality weighting architectures.}\n")
        f.write("\\label{tab:benchmark_results}\n\\end{table}\n")

    # FIGURE 3: Bar Chart of Model Coherence across Alphas (for 224px as standard reference)
    print("\nGenerating Figure 3 (Model & Alpha Sensitivity)...")
    plt.figure(figsize=(10, 6))
    df_chart = df_res[df_res['Patch_Size'].isin(['224', '336', 'N/A'])].copy()
    
    # Group by Model and Alpha for patch size 224
    df_224 = df_res[(df_res['Patch_Size'] == '224') | (df_res['Model'] == 'RNA-Only')]
    
    models_list = ['RNA-Only', 'vit-base', 'phikon']
    colors = ['#7f8c8d', '#3498db', '#9b59b6']
    
    for idx, m in enumerate(models_list):
        sub_df = df_res[df_res['Model'] == m]
        if sub_df.empty:
            continue
        if m == 'RNA-Only':
            plt.axhline(sub_df['Spatial_Silhouette'].values[0], color='r', linestyle='--', label=f'RNA-Only Baseline ({sub_df["Spatial_Silhouette"].values[0]:.3f})')
        else:
            # plot max silhouette score per patch size for each model
            best_per_patch = sub_df.groupby('Patch_Size')['Spatial_Silhouette'].max()
            plt.plot(best_per_patch.index, best_per_patch.values, marker='o', linewidth=2.5, label=f'{m.upper()} (Best Alpha)')
            
    plt.title('Spatial Coherence vs. Morphological Receptive Field (Patch Size)', fontsize=14, fontweight='bold')
    plt.xlabel('Patch Size (pixels)', fontsize=12)
    plt.ylabel('Spatial Silhouette Score (Higher = More Coherent)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('results/fig3_model_alpha_comparison.png', dpi=300)
    plt.close()
    
    # FIGURE 4: Spatial Visual Comparison Grid (RNA vs Best ViT vs Best Phikon)
    print("Generating Figure 4 (Spatial Domain Grid Comparison)...")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    sq.pl.spatial_scatter(adata, color='rna_leiden', ax=axs[0], title=f"RNA-Only Baseline\n(Sil: {sil_rna:.3f})")
    if best_vit_key and best_vit_key in adata.obs:
        sq.pl.spatial_scatter(adata, color=best_vit_key, ax=axs[1], title=f"ViT-Base (Best Config)\n(Sil: {best_vit_score:.3f})")
    if best_phikon_key and best_phikon_key in adata.obs:
        sq.pl.spatial_scatter(adata, color=best_phikon_key, ax=axs[2], title=f"Phikon Histopathology (Best Config)\n(Sil: {best_phikon_score:.3f})")
        
    plt.tight_layout()
    plt.savefig('results/fig4_spatial_domains_grid.png', dpi=300)
    plt.close()
    
    print("\nAll extensive benchmarking experiments completed successfully!")
    print("Check the 'results/' directory for generated CSVs, LaTeX tables, and plots.")

if __name__ == '__main__':
    main()
