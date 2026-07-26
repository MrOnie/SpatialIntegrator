import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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
    print("GENERATING BIOMARKER VALIDATION FIGURE (224px RESOLUTION)")
    print("="*70)
    
    print("\n1. Loading Visium H&E dataset from Squidpy...")
    adata = sq.datasets.visium_hne_adata()
    
    print("2. Preprocessing RNA modality...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    
    adata_rna = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_rna)
    sc.tl.pca(adata_rna)
    
    # Initialize SpatialIntegrator
    library_id = list(adata.uns['spatial'].keys())[0]
    dataset = SpatialDataset(adata_rna, library_id=library_id)
    fuser = ModalityFuser(n_components=50)
    
    rna_matrix = dataset.adata.X
    if hasattr(rna_matrix, 'toarray'):
        rna_matrix = rna_matrix.toarray()
        
    patch_size = 224
    alpha = 0.5
    
    print(f"3. Extracting {patch_size}x{patch_size}px morphological patches...")
    patches = dataset.extract_patches(patch_size=patch_size)
    
    # Use Phikon if available, otherwise fallback to vit-base
    model_name = 'phikon'
    try:
        print(f"4. Initializing pathology foundation model: {model_name}...")
        embedder = ImageEmbedder(model_name=model_name)
    except Exception as e:
        print(f"Could not initialize {model_name} ({e}). Using vit-base...")
        model_name = 'vit-base'
        embedder = ImageEmbedder(model_name=model_name)
        
    print("5. Computing Vision Embeddings...")
    img_embeddings = embedder.extract_embeddings(patches, batch_size=32)
    
    print("6. Fusing modalities and running Leiden clustering (igraph backend)...")
    joint_rep = fuser.fit_transform(rna_matrix, img_embeddings, alpha=alpha)
    key_added = f'multimodal_{model_name}_{patch_size}'
    
    adata_res = cluster_multimodal(dataset.adata, joint_rep, key_added=key_added, resolution=1.0)
    
    sil_score = silhouette_score(adata_res.obsm['spatial'], adata_res.obs[key_added])
    print(f" -> Resulting Spatial Silhouette Score: {sil_score:.4f} ({len(adata_res.obs[key_added].unique())} domains)")
    
    print("7. Calculating differentially expressed gene biomarkers (Wilcoxon rank-sum test)...")
    rank_key = f'rank_genes_{model_name}_{patch_size}'
    sc.tl.rank_genes_groups(adata_res, key_added, method='wilcoxon', key_added=rank_key)
    
    print("8. Plotting Biomarker Dotplot (Figure 3)...")
    # Set up high resolution figure formatting
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    dp = sc.pl.rank_genes_groups_dotplot(
        adata_res,
        n_genes=4,
        key=rank_key,
        show=False,
        title=f"Top Biomarkers per Domain ({model_name.upper()} 224px Multimodal Fusion)",
        return_fig=True
    )
    
    dp.savefig('results/fig3_biomarker_deg_validation_dotplot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Figure successfully saved to results/fig3_biomarker_deg_validation_dotplot.png!")

if __name__ == '__main__':
    main()
