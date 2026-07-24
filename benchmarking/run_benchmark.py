import os
import sys
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from sklearn.metrics import silhouette_score
import pandas as pd

# Add local package to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatialintegrator.core.dataset import SpatialDataset
from spatialintegrator.models.vision_extractor import ImageEmbedder
from spatialintegrator.models.fusion import ModalityFuser
from spatialintegrator.tl.clustering import cluster_multimodal

def main():
    os.makedirs('results', exist_ok=True)
    print("Downloading/Loading Visium H&E dataset from Squidpy...")
    # This downloads a public 10x Genomics breast cancer dataset
    adata = sq.datasets.visium_hne_adata()
    
    print("Preprocessing RNA data...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    
    adata_rna = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_rna)
    sc.tl.pca(adata_rna)
    
    # 1. CLASSICAL CLUSTERING (RNA ONLY)
    print("Running classical RNA clustering...")
    sc.pp.neighbors(adata_rna, use_rep='X_pca', key_added='rna_neighbors')
    sc.tl.leiden(adata_rna, resolution=1.0, neighbors_key='rna_neighbors', key_added='rna_leiden')
    
    sil_rna = silhouette_score(adata_rna.obsm['spatial'], adata_rna.obs['rna_leiden'])
    print(f"-> Spatial Silhouette Score (RNA Only): {sil_rna:.4f}")
    
    # RNA Biomarkers
    sc.tl.rank_genes_groups(adata_rna, 'rna_leiden', method='t-test', key_added='rank_genes_rna')
    adata.obs['rna_leiden'] = adata_rna.obs['rna_leiden']
    
    # Initialize SpatialIntegrator
    print("\nInitializing SpatialIntegrator...")
    library_id = list(adata.uns['spatial'].keys())[0]
    dataset = SpatialDataset(adata_rna, library_id=library_id)
    
    # Note: For benchmarking, we use vit-base by default. 
    # Change model_name='uni' if you have access and want to test pathology models.
    embedder = ImageEmbedder(model_name='vit-base') 
    fuser = ModalityFuser(n_components=50)
    
    rna_matrix = dataset.adata.X
    if hasattr(rna_matrix, 'toarray'):
        rna_matrix = rna_matrix.toarray()
        
    scores = {'RNA': sil_rna}
    
    for patch_size in [112, 224]:
        print(f"\n--- Running Multimodal Pipeline with patch_size = {patch_size} ---")
        
        print("Extracting image patches...")
        patches = dataset.extract_patches(patch_size=patch_size)
        
        print("Computing Vision Embeddings...")
        img_embeddings = embedder.extract_embeddings(patches, batch_size=32)
        
        print("Fusing modalities...")
        joint_rep = fuser.fit_transform(rna_matrix, img_embeddings, alpha=0.5)
        
        key_added = f'multimodal_leiden_{patch_size}'
        print("Clustering in joint space...")
        adata_res = cluster_multimodal(dataset.adata, joint_rep, key_added=key_added, resolution=1.0)
        
        # Evaluate Spatial Coherence
        sil_multi = silhouette_score(adata_res.obsm['spatial'], adata_res.obs[key_added])
        scores[f'Multi_{patch_size}'] = sil_multi
        print(f"-> Spatial Silhouette Score (Multimodal {patch_size}px): {sil_multi:.4f}")
        
        # Find marker genes for these new clusters
        print("Calculating differentially expressed genes (Biomarkers)...")
        sc.tl.rank_genes_groups(adata_res, key_added, method='t-test', key_added=f'rank_genes_{patch_size}')
        
        adata.obs[key_added] = adata_res.obs[key_added]

    # PLOT SPATIAL COMPARISON
    print("\nGenerating Comparative Figures...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    sq.pl.spatial_scatter(adata, color='rna_leiden', ax=axs[0], title=f"RNA (Sil: {scores['RNA']:.3f})")
    sq.pl.spatial_scatter(adata, color='multimodal_leiden_112', ax=axs[1], title=f"Multi-112px (Sil: {scores['Multi_112']:.3f})")
    sq.pl.spatial_scatter(adata, color='multimodal_leiden_224', ax=axs[2], title=f"Multi-224px (Sil: {scores['Multi_224']:.3f})")
    plt.tight_layout()
    plt.savefig('results/figura_1_benchmarking.png', dpi=300)
    
    # PLOT GENE MARKERS (For the 224px approach)
    print("Generating Gene Markers Figure...")
    # Plot top 5 genes per cluster
    sc.pl.rank_genes_groups_dotplot(adata_res, n_genes=5, key='rank_genes_224', show=False)
    plt.savefig('results/figura_2_marcadores_224.png', bbox_inches='tight')
    
    print("\nBenchmark finished! Results saved in the 'results/' folder.")

if __name__ == '__main__':
    main()
