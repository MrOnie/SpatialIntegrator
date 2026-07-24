import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ModalityFuser:
    """
    Fuses RNA expression data and morphological Image embeddings into a joint latent space.
    """
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca_rna = PCA(n_components=n_components)
        self.pca_img = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
    def fit_transform(self, rna_matrix: np.ndarray, img_embeddings: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Creates a joint multimodal representation via weighted concatenation of reduced spaces.
        
        Args:
            rna_matrix: The gene expression matrix (n_spots, n_genes).
            img_embeddings: The image embeddings (n_spots, hidden_size).
            alpha: Weight for the RNA modality (0 to 1). (1-alpha) is the weight for the Image modality.
            
        Returns:
            A numpy array of the joint representation.
        """
        if rna_matrix.shape[0] != img_embeddings.shape[0]:
            raise ValueError("RNA matrix and image embeddings must have the same number of observations (spots).")
            
        # Standardize features
        rna_scaled = self.scaler.fit_transform(rna_matrix)
        img_scaled = self.scaler.fit_transform(img_embeddings)
        
        # PCA reduction on both to match dimensionality
        if rna_scaled.shape[1] > self.n_components:
            rna_reduced = self.pca_rna.fit_transform(rna_scaled)
        else:
            rna_reduced = rna_scaled
            
        if img_scaled.shape[1] > self.n_components:
            img_reduced = self.pca_img.fit_transform(img_scaled)
        else:
            img_reduced = img_scaled
            
        # Weighted concatenation
        joint_representation = np.hstack([alpha * rna_reduced, (1 - alpha) * img_reduced])
        
        return joint_representation
