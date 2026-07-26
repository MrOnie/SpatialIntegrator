import os
import tempfile
import pytest
from spatialintegrator.core.dataset import SpatialDataset
from tests.data.generate_synthetic_hd import generate_synthetic_visium_hd, generate_synthetic_xenium

def test_visium_hd_loader_and_preprocessing():
    with tempfile.TemporaryDirectory(prefix="test_hd_") as tmpdir:
        hd_dir, adata_path = generate_synthetic_visium_hd(tmpdir, bin_size=16, n_bins=80, n_genes=120)
        
        # Test directory load
        ds = SpatialDataset.from_visium_hd(tmpdir, bin_size_um=16)
        assert ds.adata.uns['modality_type'] == 'visium_hd_16um'
        assert ds.adata.n_obs == 80
        
        # Test patch extraction
        patches = ds.extract_patches(patch_size=64, res='hires')
        assert patches.shape == (80, 64, 64, 3)
        
        # Test preprocessing
        ds.preprocess_rna(n_top_genes=50)
        assert 'X_pca' in ds.adata.obsm
        assert ds.adata.obsm['X_pca'].shape[0] == 80

def test_xenium_loader_and_preprocessing():
    with tempfile.TemporaryDirectory(prefix="test_xenium_") as tmpdir:
        xenium_dir, _ = generate_synthetic_xenium(tmpdir, n_cells=60, n_genes=80)
        
        ds = SpatialDataset.from_xenium(xenium_dir)
        assert ds.adata.uns['modality_type'] == 'xenium_subcellular'
        assert ds.adata.n_obs == 60
        
        patches = ds.extract_patches(patch_size=48, res='hires')
        assert patches.shape == (60, 48, 48, 3)
        
        ds.preprocess_rna(n_top_genes=40)
        assert 'X_pca' in ds.adata.obsm
        assert ds.adata.obsm['X_pca'].shape[0] == 60
