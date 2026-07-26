import numpy as np
import pytest
from spatialintegrator.models.fusion import ModalityFuser

def test_static_vs_adaptive_fusion():
    np.random.seed(42)
    n_spots = 100
    rna_matrix = np.random.rand(n_spots, 200)
    img_embeddings = np.random.rand(n_spots, 384)
    
    fuser = ModalityFuser(n_components=20)
    
    # 1. Test static fusion (V1 regression invariance)
    joint_static = fuser.fit_transform(rna_matrix, img_embeddings, alpha=0.3, adaptive=False)
    assert joint_static.shape == (n_spots, 40)
    static_weights = fuser.get_adaptive_weights()
    assert np.allclose(static_weights, 0.3)
    
    # 2. Test Spatially Adaptive Modality Dominance (α_i)
    joint_adaptive = fuser.fit_transform(rna_matrix, img_embeddings, alpha=0.5, adaptive=True, gain=0.3)
    assert joint_adaptive.shape == (n_spots, 40)
    adaptive_weights = fuser.get_adaptive_weights()
    
    # Verify weights are heterogeneous across spots and bounded in [0.05, 0.95]
    assert adaptive_weights.shape == (n_spots,)
    assert np.min(adaptive_weights) >= 0.05
    assert np.max(adaptive_weights) <= 0.95
    assert np.std(adaptive_weights) > 0.0, "Adaptive weights must reflect local spatial entropy variations!"
    
def test_modality_fuser_small_sample_fallback():
    # Test boundary condition where samples < n_components
    rna_matrix = np.random.rand(5, 50)
    img_embeddings = np.random.rand(5, 50)
    fuser = ModalityFuser(n_components=50)
    out = fuser.fit_transform(rna_matrix, img_embeddings, adaptive=True)
    assert out.shape == (5, 8)  # max_comps will be 5 - 1 = 4, so 4 + 4 = 8
