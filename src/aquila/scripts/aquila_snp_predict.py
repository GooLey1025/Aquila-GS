#!/usr/bin/env python3
"""
Aquila SNP Prediction Script

Make predictions using a trained model.

Usage:
    python aquila_snp_predict.py --checkpoint best_checkpoint.pt --geno data/test.geno --output predictions.csv
"""

import argparse
import torch
import pandas as pd
from pathlib import Path

from aquila.varnn import VariantsNeuralNetwork
# Backward compatibility
SNPNeuralNetwork = VariantsNeuralNetwork
from aquila.data_utils import parse_genotype_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with Aquila SNP model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--geno',
        type=str,
        required=True,
        help='Path to genotype file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output CSV file for predictions'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("=" * 80)
    print("AQUILA: SNP Neural Network Prediction")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Get model configuration from checkpoint if available
    if 'config' in checkpoint:
        model_config = checkpoint['config'].get('model', {})
    else:
        print("Warning: No config found in checkpoint. Using default configuration.")
        model_config = {}
    
    # Load genotype data
    print(f"\nLoading genotype data: {args.geno}")
    snp_matrix, sample_ids, snp_ids = parse_genotype_file(args.geno)
    seq_length = snp_matrix.shape[1]
    
    print(f"Found {len(sample_ids)} samples with {seq_length} SNPs")
    
    # Convert to tensor
    snp_tensor = torch.from_numpy(snp_matrix).long()
    
    # Create model (you'll need to match the architecture used during training)
    # This is a simplified version - in practice, you'd save model config in checkpoint
    print("\nCreating model...")
    
    # Try to extract task information from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Infer number of tasks from output layers
        num_regression_tasks = 0
        num_classification_tasks = 0
        regression_tasks = []
        classification_tasks = []
        
        for key in state_dict.keys():
            if 'regression_head' in key and 'network' in key:
                # Find the last linear layer
                if key.endswith('.weight'):
                    num_regression_tasks = state_dict[key].shape[0]
                    regression_tasks = [f"regression_task_{i}" for i in range(num_regression_tasks)]
            elif 'classification_head' in key and 'network' in key:
                if key.endswith('.weight'):
                    num_classification_tasks = state_dict[key].shape[0]
                    classification_tasks = [f"classification_task_{i}" for i in range(num_classification_tasks)]
        
        print(f"Detected {num_regression_tasks} regression tasks")
        print(f"Detected {num_classification_tasks} classification tasks")
    
    # Create model with detected configuration
    model = SNPNeuralNetwork(
        seq_length=seq_length,
        embed_dim=model_config.get('embed_dim', 128),
        num_transformer_layers=model_config.get('num_transformer_layers', 4),
        num_heads=model_config.get('num_heads', 8),
        d_ff=model_config.get('d_ff', 512),
        dropout=0.0,  # No dropout during inference
        trunk_type=model_config.get('trunk_type', 'transformer'),
        pool_type=model_config.get('pool_type', 'attention'),
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks,
        regression_hidden_dim=model_config.get('regression_hidden_dim', 256),
        classification_hidden_dim=model_config.get('classification_hidden_dim', 256),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Make predictions
    print("\nMaking predictions...")
    all_predictions = {'regression': [], 'classification': []}
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(snp_tensor), args.batch_size):
            batch = snp_tensor[i:i+args.batch_size].to(args.device)
            outputs = model(batch)
            
            if 'regression' in outputs:
                all_predictions['regression'].append(outputs['regression'].cpu().numpy())
            if 'classification' in outputs:
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs['classification'])
                all_predictions['classification'].append(probs.cpu().numpy())
    
    # Concatenate all batches
    import numpy as np
    results_df = pd.DataFrame({'sample_id': sample_ids})
    
    if all_predictions['regression']:
        reg_preds = np.concatenate(all_predictions['regression'], axis=0)
        for i in range(reg_preds.shape[1]):
            results_df[f'regression_task_{i}'] = reg_preds[:, i]
    
    if all_predictions['classification']:
        cls_preds = np.concatenate(all_predictions['classification'], axis=0)
        for i in range(cls_preds.shape[1]):
            results_df[f'classification_task_{i}_prob'] = cls_preds[:, i]
            results_df[f'classification_task_{i}_pred'] = (cls_preds[:, i] >= 0.5).astype(int)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total samples: {len(results_df)}")
    print("\nPreview:")
    print(results_df.head())
    
    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

