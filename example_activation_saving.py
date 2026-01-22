"""
Example script demonstrating how to use nnUNetTrainer with activation saving enabled.

This script shows:
1. How to enable activation saving
2. How to customize the save frequency
3. How to load and analyze saved activations
"""

import torch
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json
import os


def enable_activation_saving_in_trainer(trainer, save_frequency=10):
    """
    Enable activation saving in an existing trainer instance.
    
    Args:
        trainer: Instance of nnUNetTrainer
        save_frequency: Save activations every N epochs (default: 10)
    """
    trainer.save_activations = True
    trainer.activation_save_frequency = save_frequency
    print(f"Activation saving enabled. Will save every {save_frequency} epochs.")


def load_saved_activations(activation_folder, epoch):
    """
    Load saved activations for a specific epoch.
    
    Args:
        activation_folder: Path to the activations folder
        epoch: Epoch number to load
    
    Returns:
        Dictionary containing the loaded activations
    """
    epoch_folder = join(activation_folder, f'epoch_{epoch}')
    
    if not os.path.exists(epoch_folder):
        print(f"No activations found for epoch {epoch}")
        return None
    
    # Load metadata
    metadata_path = join(epoch_folder, 'metadata.json')
    if os.path.exists(metadata_path):
        metadata = load_json(metadata_path)
        print(f"Metadata: {metadata}")
    
    # Load activation files
    activations = {}
    for filename in os.listdir(epoch_folder):
        if filename.endswith('.npy'):
            layer_name = filename.replace('.npy', '')
            activation_path = join(epoch_folder, filename)
            activations[layer_name] = np.load(activation_path)
            print(f"Loaded {layer_name}: shape {activations[layer_name].shape}")
    
    return activations


def analyze_activations(activations):
    """
    Perform basic analysis on saved activations.
    
    Args:
        activations: Dictionary of activation arrays
    """
    print("\n=== Activation Analysis ===")
    for layer_name, activation in activations.items():
        print(f"\nLayer: {layer_name}")
        print(f"  Shape: {activation.shape}")
        print(f"  Mean: {activation.mean():.4f}")
        print(f"  Std: {activation.std():.4f}")
        print(f"  Min: {activation.min():.4f}")
        print(f"  Max: {activation.max():.4f}")


# Example usage in your training script:
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ... setup your trainer as usual ...
trainer = nnUNetTrainer(plans, configuration, fold, dataset_json, device)

# Enable activation saving
trainer.save_activations = True
trainer.activation_save_frequency = 10  # Save every 10 epochs

# Run training
trainer.run_training()

# After training, load and analyze activations
activations_folder = join(trainer.output_folder, 'activations')
activations = load_saved_activations(activations_folder, epoch=50)
if activations is not None:
    analyze_activations(activations)
"""


if __name__ == "__main__":
    print("This is an example script for activation saving.")
    print("Import the functions in your training script to enable activation saving.")
    print("\nExample:")
    print("  from example_activation_saving import enable_activation_saving_in_trainer")
    print("  enable_activation_saving_in_trainer(trainer, save_frequency=5)")
