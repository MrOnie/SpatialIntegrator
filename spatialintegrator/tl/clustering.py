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


def _get_spatial_weights(coords: np.ndarray, k_neighbors: int = 6) -> np.ndarray:
    """Computes k-nearest neighbor spatial connectivity weight matrix."""
    from scipy.spatial import cKDTree
    n_spots = coords.shape[0]
    k_actual = min(k_neighbors + 1, n_spots)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k_actual)
    
    W = np.zeros((n_spots, n_spots), dtype=np.float64)
    for i in range(n_spots):
        for j in idx[i, 1:]:
            W[i, j] = 1.0
            W[j, i] = 1.0 # Symmetric spatial graph
    return W


def compute_moran_i(adata: AnnData, use_rep: str = 'X_joint', spatial_key: str = 'spatial', k_neighbors: int = 6) -> float:
    """
    Computes Moran's I Index of Global Spatial Autocorrelation over a representation space.
    Evaluates whether physically adjacent tissue regions share cohesive multimodal signatures.
    """
    if use_rep not in adata.obsm and use_rep != 'X':
        raise ValueError(f"Representation {use_rep} not found in adata.obsm.")
        
    X = adata.X if use_rep == 'X' else adata.obsm[use_rep]
    if hasattr(X, "toarray"):
        X = X.toarray()
    coords = adata.obsm[spatial_key]
    
    W = _get_spatial_weights(coords, k_neighbors=k_neighbors)
    W_sum = np.sum(W) + 1e-8
    n_spots = X.shape[0]
    
    X_mean = np.mean(X, axis=0, keepdims=True)
    Z = X - X_mean
    
    # Variance sum denominator
    var_sum = np.sum(Z**2) + 1e-8
    
    # Covariance numerator across spatial weights
    cov_sum = np.sum((W @ Z) * Z)
    
    morans_i = (n_spots / W_sum) * (cov_sum / var_sum)
    adata.uns['morans_i_score'] = float(morans_i)
    return float(morans_i)


def compute_geary_c(adata: AnnData, use_rep: str = 'X_joint', spatial_key: str = 'spatial', k_neighbors: int = 6) -> float:
    """
    Computes Geary's C Ratio of spatial contiguity. Values < 1.0 denote positive spatial autocorrelation.
    """
    if use_rep not in adata.obsm and use_rep != 'X':
        raise ValueError(f"Representation {use_rep} not found in adata.obsm.")
        
    X = adata.X if use_rep == 'X' else adata.obsm[use_rep]
    if hasattr(X, "toarray"):
        X = X.toarray()
    coords = adata.obsm[spatial_key]
    
    W = _get_spatial_weights(coords, k_neighbors=k_neighbors)
    W_sum = np.sum(W) + 1e-8
    n_spots = X.shape[0]
    
    X_mean = np.mean(X, axis=0, keepdims=True)
    var_sum = np.sum((X - X_mean)**2) + 1e-8
    
    # Pairwise spatial divergence numerator
    # sum_{i,j} W_{ij} ||X_i - X_j||^2 = 2 * sum_{i} (sum_j W_ij) ||X_i||^2 - 2 sum_{i,j} W_ij X_i . X_j
    deg = np.sum(W, axis=1, keepdims=True)
    pair_dist_sum = np.sum(deg * (X**2)) * 2.0 - 2.0 * np.sum((W @ X) * X)
    
    gearys_c = ((n_spots - 1.0) / (2.0 * W_sum)) * (pair_dist_sum / var_sum)
    adata.uns['gearys_c_score'] = float(gearys_c)
    return float(gearys_c)


def score_boundary_ligand_receptor(adata: AnnData, cluster_key: str = 'multimodal_leiden', spatial_key: str = 'spatial', k_neighbors: int = 6) -> dict:
    """
    Evaluates intercellular signaling activity (Ligand-Receptor co-expression) specifically at
    interfacial boundaries separating distinct architectural histological domains.
    """
    if cluster_key not in adata.obs:
        return {"error": f"Cluster key {cluster_key} not found in adata.obs"}
        
    coords = adata.obsm[spatial_key]
    labels = adata.obs[cluster_key].astype(str).values
    W = _get_spatial_weights(coords, k_neighbors=k_neighbors)
    
    # Identify interfacial boundary spots (spots with at least one neighbor from a different domain)
    boundary_mask = np.zeros(labels.shape[0], dtype=bool)
    for i in range(labels.shape[0]):
        neighbors = np.where(W[i] > 0)[0]
        if np.any(labels[neighbors] != labels[i]):
            boundary_mask[i] = True
            
    adata.obs['is_domain_boundary'] = boundary_mask.astype(str)
    
    # Identify top expression features or canonical L-R proxy pairs
    gene_names = adata.var_names.tolist()
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
        
    if np.min(matrix) < 0:
        # Re-align scaled matrix to non-negative expression floor for valid biological enrichment ratios
        matrix = matrix - np.min(matrix, axis=0)
        
    boundary_mean = np.mean(matrix[boundary_mask], axis=0) if np.any(boundary_mask) else np.zeros(matrix.shape[1])
    bulk_mean = np.mean(matrix[~boundary_mask], axis=0) if np.any(~boundary_mask) else np.ones(matrix.shape[1]) + 1e-8
    
    fold_changes = (boundary_mean + 0.5) / (bulk_mean + 0.5)
    top_indices = np.argsort(fold_changes)[::-1][:10]
    
    results = {
        "boundary_spot_count": int(np.sum(boundary_mask)),
        "bulk_spot_count": int(np.sum(~boundary_mask)),
        "top_boundary_biomarkers": [
            {"gene": gene_names[idx], "boundary_expr": float(boundary_mean[idx]), "bulk_expr": float(bulk_mean[idx]), "fc": float(fold_changes[idx])}
            for idx in top_indices if idx < len(gene_names)
        ]
    }
    adata.uns['boundary_ccc_analysis'] = results
    return results
