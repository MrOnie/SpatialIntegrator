import numpy as np
import pandas as pd
import anndata as ad
import pytest
from spatialintegrator.tl.clustering import compute_moran_i, compute_geary_c, score_boundary_ligand_receptor

@pytest.fixture
def sample_spatial_adata():
    np.random.seed(101)
    n_spots = 100
    # Create smooth gradient across X coordinates to force strong positive spatial autocorrelation
    coords = np.zeros((n_spots, 2))
    coords[:, 0] = np.linspace(0, 10, n_spots)
    coords[:, 1] = np.sin(np.linspace(0, 5, n_spots)) * 2.0
    
    # Feature matrix correlated with X coordinate
    feature = np.repeat(coords[:, 0:1], 10, axis=1) + np.random.normal(0, 0.1, (n_spots, 10))
    
    obs = pd.DataFrame({'multimodal_leiden': ['0' if x < 5 else '1' for x in coords[:, 0]]}, index=[f"spot_{i}" for i in range(n_spots)])
    var = pd.DataFrame(index=[f"GENE_{j}" for j in range(10)])
    
    adata = ad.AnnData(X=feature.astype(np.float32), obs=obs, var=var)
    adata.obsm['spatial'] = coords
    adata.obsm['X_joint'] = feature.astype(np.float32)
    return adata

def test_morans_i_positive_autocorrelation(sample_spatial_adata):
    i_score = compute_moran_i(sample_spatial_adata, use_rep='X_joint', k_neighbors=4)
    # Smooth continuous spatial gradients must produce positive Moran's I (> 0.5)
    assert i_score > 0.5, f"Expected high positive Moran's I, got {i_score}"
    assert 'morans_i_score' in sample_spatial_adata.uns

def test_gearys_c_positive_contiguity(sample_spatial_adata):
    c_score = compute_geary_c(sample_spatial_adata, use_rep='X_joint', k_neighbors=4)
    # Geary's C < 1.0 denotes smooth positive spatial autocorrelation
    assert c_score < 0.5, f"Expected low Geary's C (<1) for smooth gradients, got {c_score}"
    assert 'gearys_c_score' in sample_spatial_adata.uns

def test_boundary_ccc_enrichment(sample_spatial_adata):
    res = score_boundary_ligand_receptor(sample_spatial_adata, cluster_key='multimodal_leiden', k_neighbors=4)
    assert 'boundary_spot_count' in res
    assert res['boundary_spot_count'] > 0
    assert len(res['top_boundary_biomarkers']) == 10
    assert 'is_domain_boundary' in sample_spatial_adata.obs
