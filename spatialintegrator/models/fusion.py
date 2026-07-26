import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ModalityFuser:
    """
    Fuses RNA expression data and morphological Image embeddings into a joint latent space.
    Supports both invariant global weighting (α) and Spatially Adaptive Modality Dominance (α_i).
    """
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca_rna = PCA(n_components=n_components)
        self.pca_img = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.last_alpha_weights = None
        
    def compute_adaptive_alpha(self, rna_eq: np.ndarray, img_eq: np.ndarray, base_alpha: float = 0.5, gain: float = 0.3) -> np.ndarray:
        """
        Computes Spatially Adaptive Modality Dominance weights α_i for each anatomical spot/bin.
        
        Derives local weight based on transcriptional Shannon entropy (information density)
        vs. morphological textural distinctiveness from the spatial foundation model.
        """
        eps = 1e-8
        n_spots, dims_rna = rna_eq.shape
        _, dims_img = img_eq.shape
        
        # 1. Transcriptional Specificity (1 - normalized Shannon Entropy over expression components)
        # Convert latent feature intensities into local probability distribution p_ij
        rna_abs = np.abs(rna_eq) + eps
        p_ij = rna_abs / np.sum(rna_abs, axis=1, keepdims=True)
        shannon_entropy = -np.sum(p_ij * np.log(p_ij + eps), axis=1)
        max_entropy = np.log(dims_rna + eps)
        specificity_rna = 1.0 - (shannon_entropy / max_entropy)
        
        # 2. Morphological Textural Distinctiveness (Euclidean deviation from mean tissue architecture)
        img_mean = np.mean(img_eq, axis=0, keepdims=True)
        dev_img = np.linalg.norm(img_eq - img_mean, axis=1)
        
        # Min-max normalize both informational signals across the entire whole-slide slice
        norm_rna = (specificity_rna - np.min(specificity_rna)) / (np.max(specificity_rna) - np.min(specificity_rna) + eps)
        norm_img = (dev_img - np.min(dev_img)) / (np.max(dev_img) - np.min(dev_img) + eps)
        
        # 3. Derive dynamic spatial balancing parameter α_i around base_alpha
        delta_signal = norm_rna - norm_img
        alpha_i = base_alpha + gain * delta_signal
        
        # Clip weights to ensure numerical stability and preserve baseline multimodal synergy
        alpha_i = np.clip(alpha_i, 0.05, 0.95).reshape(-1, 1)
        return alpha_i
        
    def fit_transform(self, rna_matrix: np.ndarray, img_embeddings: np.ndarray, alpha: float = 0.5, adaptive: bool = False, gain: float = 0.3) -> np.ndarray:
        """
        Creates a joint multimodal representation via weighted concatenation of reduced spaces.
        
        Args:
            rna_matrix: The gene expression matrix (n_spots, n_genes).
            img_embeddings: The image embeddings (n_spots, hidden_size).
            alpha: Weight for the RNA modality (0 to 1) or base alpha when adaptive=True.
            adaptive: If True, engages Spatially Adaptive Modality Dominance (α_i per spot).
            gain: Sensitivity gain factor γ for adaptive weighting (default 0.3).
            
        Returns:
            A numpy array of the joint representation (n_spots, n_components * 2).
        """
        if rna_matrix.shape[0] != img_embeddings.shape[0]:
            raise ValueError("RNA matrix and image embeddings must have the same number of observations (spots).")
            
        # Standardize features
        rna_scaled = self.scaler.fit_transform(rna_matrix)
        img_scaled = self.scaler.fit_transform(img_embeddings)
        
        # PCA reduction on both to match dimensionality
        max_comps = min(self.n_components, rna_scaled.shape[0] - 1, rna_scaled.shape[1], img_scaled.shape[1])
        if max_comps < 1:
            max_comps = 1
            
        if rna_scaled.shape[1] > max_comps:
            self.pca_rna = PCA(n_components=max_comps)
            rna_reduced = self.pca_rna.fit_transform(rna_scaled)
        else:
            rna_reduced = rna_scaled
            
        if img_scaled.shape[1] > max_comps:
            self.pca_img = PCA(n_components=max_comps)
            img_reduced = self.pca_img.fit_transform(img_scaled)
        else:
            img_reduced = img_scaled
            
        # Equalize spectral total variance (Frobenius norm inertia equalization)
        norm_r = np.linalg.norm(rna_reduced, ord='fro') + 1e-8
        norm_v = np.linalg.norm(img_reduced, ord='fro') + 1e-8
        target_norm = np.sqrt(rna_matrix.shape[0] * max_comps)
        
        rna_eq = (rna_reduced / norm_r) * target_norm
        img_eq = (img_reduced / norm_v) * target_norm
            
        # Determine global vs. spatially adaptive weights α_i
        if adaptive:
            alpha_weights = self.compute_adaptive_alpha(rna_eq, img_eq, base_alpha=alpha, gain=gain)
        else:
            alpha_weights = np.full((rna_matrix.shape[0], 1), alpha, dtype=np.float64)
            
        self.last_alpha_weights = alpha_weights
        
        # Weighted concatenation on algebraically equilibrated subspaces
        joint_representation = np.hstack([alpha_weights * rna_eq, (1.0 - alpha_weights) * img_eq])
        
        return joint_representation
    def get_adaptive_weights(self) -> np.ndarray:
        """Returns the array of modality weights α_i applied during the last fit_transform call."""
        if self.last_alpha_weights is None:
            raise RuntimeError("fit_transform has not been executed yet.")
        return self.last_alpha_weights.flatten()
        
    def get_last_alpha_weights(self) -> np.ndarray:
        """Alias for get_adaptive_weights."""
        return self.get_adaptive_weights()
