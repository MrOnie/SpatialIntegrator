import scanpy as sc
from anndata import AnnData
import numpy as np

def cluster_multimodal(adata: AnnData, joint_representation: np.ndarray, key_added: str = 'multimodal_leiden', resolution: float = 1.0):
    """
    Performs clustering on the joint multimodal representation.
    
    Args:
        adata: The AnnData object.
        joint_representation: The fused latent space array.
        key_added: The key under which to save the cluster labels in `adata.obs`.
        resolution: Resolution parameter for the Leiden algorithm.
        
    Returns:
        The updated AnnData object.
    """
    # Store the joint representation in obsm
    adata.obsm['X_joint'] = joint_representation
    
    # Compute neighborhood graph based on joint representation
    sc.pp.neighbors(adata, use_rep='X_joint', key_added='joint')
    
    # Run UMAP for 2D visualization of the joint space
    sc.tl.umap(adata, neighbors_key='joint')
    
    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=resolution, neighbors_key='joint', key_added=key_added, flavor='igraph', n_iterations=2, directed=False)
    
    return adata
