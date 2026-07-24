import scanpy as sc
import squidpy as sq
from anndata import AnnData
import matplotlib.pyplot as plt

def plot_spatial_domains(adata: AnnData, color: str = 'multimodal_leiden', title: str = "Multimodal Spatial Domains", save_path: str = None):
    """
    Plots the spatial clusters overlaid on the H&E image.
    
    Args:
        adata: The AnnData object with spatial coordinates and images.
        color: The column in `adata.obs` that contains the cluster labels.
        title: Title of the plot.
        save_path: Optional path to save the generated figure.
        
    Returns:
        The matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sq.pl.spatial_scatter(
        adata,
        color=color,
        shape="circle",
        alpha=0.7,
        size=1.5,
        ax=ax,
        title=title
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_joint_umap(adata: AnnData, color: str = 'multimodal_leiden', save_path: str = None):
    """
    Plots the UMAP of the joint multimodal space.
    """
    fig = sc.pl.umap(adata, color=color, title="UMAP of Joint Vision-RNA Space", show=False, return_fig=True)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
    return fig
