import scanpy as sc
import squidpy as sq
from anndata import AnnData
import numpy as np
import os

class SpatialDataset:
    """
    A wrapper class for AnnData to handle spatial transcriptomics data and H&E images.
    """
    def __init__(self, adata: AnnData, library_id: str = None):
        self.adata = adata
        if 'spatial' not in self.adata.uns:
            raise ValueError("The AnnData object does not contain 'spatial' data in `.uns`.")
            
        if library_id is None:
            self.library_id = list(self.adata.uns['spatial'].keys())[0]
        else:
            self.library_id = library_id
            
    @classmethod
    def from_visium(cls, path: str, library_id: str = None):
        """
        Loads a 10x Genomics Visium dataset from a directory.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        adata = sc.read_visium(path, library_id=library_id)
        # Calculate basic QC
        adata.var_names_make_unique()
        return cls(adata, library_id)

    def get_image(self, res: str = 'hires') -> np.ndarray:
        """Returns the H&E image at the specified resolution."""
        return self.adata.uns['spatial'][self.library_id]['images'][res]
        
    def get_scalefactor(self, res: str = 'hires') -> float:
        """Returns the scale factor for the specified resolution."""
        return self.adata.uns['spatial'][self.library_id]['scalefactors'][f'tissue_{res}_scalef']
        
    def extract_patches(self, patch_size: int = 224, res: str = 'hires') -> np.ndarray:
        """
        Extracts image patches centered around each spatial spot.
        The extracted patches are stored in `self.adata.obsm['patches']`.
        
        Args:
            patch_size: Target size of the patch (width and height).
            res: Resolution of the image to use ('hires' or 'lowres').
            
        Returns:
            A numpy array of shape (n_obs, patch_size, patch_size, channels).
        """
        img = self.get_image(res)
        scale = self.get_scalefactor(res)
        
        # Coordinates are usually [x, y] in AnnData
        coords = self.adata.obsm['spatial']
        scaled_coords = coords * scale
        
        patches = []
        half_size = patch_size // 2
        
        for coord in scaled_coords:
            x, y = int(coord[0]), int(coord[1])
            
            y_start = max(0, y - half_size)
            y_end = min(img.shape[0], y + half_size + (patch_size % 2))
            x_start = max(0, x - half_size)
            x_end = min(img.shape[1], x + half_size + (patch_size % 2))
            
            patch = img[y_start:y_end, x_start:x_end]
            
            # If the patch is cut off at the edges, pad it with zeros
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_y = patch_size - patch.shape[0]
                pad_x = patch_size - patch.shape[1]
                # Assuming 3 channels for H&E
                patch = np.pad(patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0)
                
            patches.append(patch)
            
        patches_array = np.array(patches)
        self.adata.obsm['patches'] = patches_array
        return patches_array

    def preprocess_rna(self, n_top_genes: int = 3000):
        """Standard preprocessing for RNA-seq data."""
        sc.pp.normalize_total(self.adata, inplace=True)
        sc.pp.log1p(self.adata)
        sc.pp.highly_variable_genes(self.adata, flavor="seurat", n_top_genes=n_top_genes)
        self.adata = self.adata[:, self.adata.var.highly_variable]
        sc.pp.scale(self.adata)
        sc.tl.pca(self.adata)
