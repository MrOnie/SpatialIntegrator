import torch
from transformers import ViTImageProcessor, ViTModel
import timm
import numpy as np
from typing import List
from huggingface_hub import login

class ImageEmbedder:
    """
    Extracts high-dimensional morphological features from image patches using Vision Foundation Models.
    Supported models: 'vit-base', 'uni', 'phikon'.
    """
    def __init__(self, model_name: str = 'vit-base', device: str = None, hf_token: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model_name = model_name
        
        # Authenticate if token is provided
        if hf_token:
            login(token=hf_token)
            
        if self.model_name == 'vit-base':
            hf_id = 'google/vit-base-patch16-224-in21k'
            self.processor = ViTImageProcessor.from_pretrained(hf_id)
            self.model = ViTModel.from_pretrained(hf_id).to(self.device)
            self.backend = 'transformers'
            
        elif self.model_name == 'phikon':
            hf_id = 'owkin/phikon'
            self.processor = ViTImageProcessor.from_pretrained(hf_id)
            self.model = ViTModel.from_pretrained(hf_id).to(self.device)
            self.backend = 'transformers'
            
        elif self.model_name == 'uni':
            hf_id = 'hf_hub:MahmoodLab/UNI'
            # timm handles preprocessing via data config
            self.model = timm.create_model(hf_id, pretrained=True, init_values=1e-5, dynamic_img_size=True).to(self.device)
            data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
            self.transform = timm.data.create_transform(**data_config, is_training=False)
            self.backend = 'timm'
            
        else:
            raise ValueError(f"Model {model_name} not supported. Use 'vit-base', 'uni', or 'phikon'.")
            
        self.model.eval()
        
    def extract_embeddings(self, patches: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extracts embeddings for a numpy array of image patches.
        
        Args:
            patches: A numpy array of shape (n_patches, H, W, C).
            batch_size: The batch size for model inference.
            
        Returns:
            A numpy array of embeddings with shape (n_patches, hidden_size).
        """
        if len(patches.shape) != 4:
            raise ValueError("Patches array must have 4 dimensions (n_patches, H, W, C).")
            
        n_patches = patches.shape[0]
        embeddings = []
        
        # Ensure patches are uint8
        if patches.dtype != np.uint8:
            if patches.max() <= 1.0:
                patches = (patches * 255).astype(np.uint8)
            else:
                patches = patches.astype(np.uint8)
                
        with torch.no_grad():
            for i in range(0, n_patches, batch_size):
                batch = patches[i:i+batch_size]
                
                if self.backend == 'transformers':
                    batch_list = [img for img in batch]
                    inputs = self.processor(images=batch_list, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    # Use the [CLS] token embedding
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embeddings)
                    
                elif self.backend == 'timm':
                    from PIL import Image
                    # Convert numpy patches to PIL images, apply transform, stack to tensor
                    tensor_list = [self.transform(Image.fromarray(img)) for img in batch]
                    inputs = torch.stack(tensor_list).to(self.device)
                    # UNI returns embedding directly
                    outputs = self.model(inputs)
                    embeddings.append(outputs.cpu().numpy())
                
        return np.vstack(embeddings)
