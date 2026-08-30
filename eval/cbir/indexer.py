"""Script for performing offline indexing of the dataset."""

import os
import sys
import argparse
import logging
import pickle
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import torch.nn as nn

CBIR_ROOT = os.path.dirname(os.path.abspath(__file__))
if CBIR_ROOT not in sys.path:
    sys.path.insert(0, CBIR_ROOT)

from models import create_model  # Unified interface

def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_model(
    model_type: str,
    model_path: str,
    device: torch.device,
    embedding_dim: int = 384,
    config_path: str = None,
    use_multi_scale: bool = False,
    layer_indices: list = None,
    img_size: int = 224,
    in_chans: int = 4
) -> nn.Module:
    """Load model using unified adapter.
    
    Args:
        model_type: Type of model ('prithvi' or 'dynamicvis' or 'prithvi2')
        model_path: Path to model checkpoint
        device: PyTorch device
        embedding_dim: Embedding dimension
        config_path: Config file path (required for DynamicVis)
        use_multi_scale: Use multi-scale features (Prithvi only)
        layer_indices: Layer indices for multi-scale (Prithvi only)
        img_size: Input image size (default: 224)
        in_chans: Number of input channels (default: 4)
    
    Returns:
        Loaded model
    """
    model = create_model(
        model_type=model_type,
        model_path=model_path,
        embedding_dim=embedding_dim,
        device=device,
        config_path=config_path,
        use_multi_scale=use_multi_scale,
        layer_indices=layer_indices,
        img_size=img_size,
        in_chans=in_chans
    )
    
    logging.info(f"Model loaded: {model_type}")
    if model_type == 'prithvi' and use_multi_scale:
        logging.info(f"Using multi-scale features from layers: {layer_indices if layer_indices else 'default [-4, -3, -2, -1]'}")
    
    return model

def extract_embeddings(model: nn.Module,
                      dataloader: DataLoader,
                      device: torch.device) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract embeddings from all tiles in the dataset."""
    embeddings = []
    metadata_list = []

    model.eval()
    with torch.no_grad():
        for batch_tiles, batch_metadata in tqdm(dataloader, desc="Extracting embeddings"):
            # Move tiles to device
            batch_tiles = batch_tiles.to(device)

            # Get embeddings
            batch_embeddings = model(batch_tiles)

            # Convert to numpy and store
            embeddings.append(batch_embeddings.cpu().numpy())

            # Handle metadata correctly
            if isinstance(batch_metadata, dict):
                # If it's a single dict with batch data
                batch_size = batch_tiles.size(0)
                for i in range(batch_size):
                    metadata_dict = {}
                    for key, value in batch_metadata.items():
                        if isinstance(value, (list, tuple)):
                            metadata_dict[key] = value[i]
                        else:
                            metadata_dict[key] = value
                    metadata_list.append(metadata_dict)
            elif isinstance(batch_metadata, (list, tuple)):
                # If it's a list of dicts
                for item in batch_metadata:
                    if isinstance(item, dict):
                        metadata_list.append(item)
                    else:
                        print(f"Warning: Expected dict, got {type(item)}: {item}")
            else:
                print(f"Error: Unexpected batch_metadata type: {type(batch_metadata)}")

    # Concatenate all embeddings
    if embeddings:
        embeddings = np.vstack(embeddings)
        logging.info(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    else:
        logging.warning("No embeddings were extracted.")
        embeddings = np.array([])
        
    logging.info(f"Metadata list length: {len(metadata_list)}")

    return embeddings, metadata_list


def build_faiss_index(embeddings: np.ndarray,
                     index_type: str = 'IVF',
                     nlist: int = 100) -> faiss.Index:
    """Build FAISS index for fast similarity search."""
    dimension = embeddings.shape[1]

    if index_type == 'Flat':
        # Exact search
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized embeddings

    elif index_type == 'IVF':
        # Approximate search with IVF
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        # Train the index
        logging.info("Training FAISS index...")
        index.train(embeddings)

    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    # Add embeddings to index
    logging.info("Adding embeddings to index...")
    index.add(embeddings)

    logging.info(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(index: faiss.Index,
                           metadata: List[Dict[str, Any]],
                           index_dir: str,
                           index_name: str = 'satellite_index') -> None:
    """Save FAISS index and metadata to disk."""
    os.makedirs(index_dir, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(index_dir, f'{index_name}.index')
    faiss.write_index(index, index_path)
    logging.info(f"FAISS index saved to: {index_path}")

    # Save metadata
    metadata_path = os.path.join(index_dir, f'{index_name}_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logging.info(f"Metadata saved to: {metadata_path}")


def get_image_files(data_dir: str) -> List[str]:
    """Get all supported image files from path."""
    image_files = []
    supported_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')

    if os.path.isfile(data_dir) and data_dir.lower().endswith(supported_exts):
        # Single file
        image_files.append(data_dir)
    elif os.path.isdir(data_dir):
        # Directory - recursively find all supported image files
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(supported_exts):
                    image_files.append(os.path.join(root, file))
    else:
        raise ValueError(f"Invalid data path: {data_dir}")

    logging.info(f"Found {len(image_files)} image files")
    return image_files


def main():
    """Main indexing function."""
    from data_loader import get_inference_loader

    parser = argparse.ArgumentParser(description='Build FAISS index for satellite imagery search')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='prithvi', 
                       choices=['prithvi', 'prithvi2', 'dynamicvis'],
                       help='Type of model to use')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config file (required for DynamicVis)')
    parser.add_argument('--embedding_dim', type=int, default=384,
                       help='Dimension of output embeddings')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to directory containing target satellite images')
    parser.add_argument('--index_dir', type=str, default='index',
                       help='Directory to save the FAISS index and metadata')
    parser.add_argument('--index_name', type=str, default='satellite_index',
                       help='Name for the index files')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for embedding extraction')
    parser.add_argument('--tile_size', type=int, default=224,
                       help='Size of tiles to extract')
    parser.add_argument('--stride', type=int, default=112,
                       help='Stride for tile extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Index arguments
    parser.add_argument('--index_type', type=str, default='IVF', 
                       choices=['Flat', 'IVF'],
                       help='Type of FAISS index to build')
    parser.add_argument('--nlist', type=int, default=100,
                       help='Number of clusters for IVF index')
    
    # Prithvi-specific arguments
    parser.add_argument('--use_multi_scale', action='store_true',
                       help='Use multi-scale features from different layers (Prithvi only)')
    parser.add_argument('--layer_indices', type=int, nargs='+', default=None,
                       help='Layer indices to use for multi-scale features (e.g., 3 6 9 11)')

    args = parser.parse_args()

    # Validate arguments
    if args.model_type == 'dynamicvis' and args.config_path is None:
        parser.error("--config_path is required when using --model_type dynamicvis")

    # Setup logging
    setup_logging()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model
    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
        embedding_dim=args.embedding_dim,
        config_path=args.config_path,
        use_multi_scale=args.use_multi_scale,
        layer_indices=args.layer_indices
    )

    # Get image files
    image_files = get_image_files(args.data_dir)

    if not image_files:
        logging.error("No supported image files found in the specified directory")
        return

    # Create data loader for inference
    dataloader = get_inference_loader(
        image_paths=image_files,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        stride=args.stride,
        num_workers=args.num_workers
    )

    logging.info(f"Processing {len(dataloader.dataset)} tiles from {len(image_files)} images")

    # Extract embeddings
    embeddings, metadata = extract_embeddings(model, dataloader, device)

    if embeddings.size == 0:
        logging.error("No embeddings were generated. Cannot build index.")
        return
        
    # Build FAISS index
    index = build_faiss_index(
        embeddings,
        index_type=args.index_type,
        nlist=args.nlist
    )

    # Save index and metadata
    save_index_and_metadata(
        index=index,
        metadata=metadata,
        index_dir=args.index_dir,
        index_name=args.index_name
    )

    # Print summary
    logging.info("Indexing completed successfully!")
    logging.info(f"Total embeddings: {embeddings.shape[0]}")
    logging.info(f"Embedding dimension: {embeddings.shape[1]}")
    logging.info(f"Index type: {args.index_type}")
    logging.info(f"Files saved in: {args.index_dir}")


if __name__ == "__main__":
    main()
