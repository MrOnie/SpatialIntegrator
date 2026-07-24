import numpy as np
import pytest
from spatialintegrator.models.fusion import ModalityFuser

def test_modality_fuser():
    # Mock data: 100 spots, 500 genes
    rna_matrix = np.random.rand(100, 500)
    # Mock data: 100 spots, 768 image embedding dim (ViT base)
    img_embeddings = np.random.rand(100, 768)
    
    fuser = ModalityFuser(n_components=20)
    joint_rep = fuser.fit_transform(rna_matrix, img_embeddings, alpha=0.5)
    
    # 20 components from RNA + 20 components from Image = 40 components
    assert joint_rep.shape == (100, 40)
    
def test_modality_fuser_shape_mismatch():
    rna_matrix = np.random.rand(100, 500)
    img_embeddings = np.random.rand(90, 768) # Mismatch
    
    fuser = ModalityFuser(n_components=20)
    with pytest.raises(ValueError):
        fuser.fit_transform(rna_matrix, img_embeddings)
