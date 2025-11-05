"""
Dynamic neural network architecture for SNP-based genomic prediction.
Builds models from YAML configuration using composable blocks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from . import blocks


class SNPNeuralNetwork(nn.Module):
    """
    Multi-task neural network for genomic prediction with dynamic architecture.
    
    The architecture is built from YAML configuration specifying:
        1. Embedding layer (SNP encoding)
        2. Trunk blocks (shared feature extraction)
        3. Head blocks (task-specific predictions)
    """
    
    def __init__(self, params: dict):
        """Initialize model from configuration parameters.
        
        Args:
            params: Configuration dictionary with:
                - seq_length: Sequence length
                - embedding: Embedding block config
                - trunk: List of trunk block configs
                - heads: Dict of head block configs (keyed by head name)
                - regression_tasks: List of regression task names
                - classification_tasks: List of classification task names
                - dropout, activation, l2_scale: Global defaults
        """
        super().__init__()
        
        self.params = params
        self.seq_length = params.get('seq_length')
        
        # Task configuration
        self.regression_tasks = params.get('regression_tasks', [])
        self.classification_tasks = params.get('classification_tasks', [])
        self.num_regression_tasks = len(self.regression_tasks)
        self.num_classification_tasks = len(self.classification_tasks)
        
        # Build model from config
        self.build_model()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def build_model(self):
        """Build model dynamically from trunk and head configs."""
        # Build embedding if specified
        if 'embedding' in self.params:
            self.embedding_block = self.build_block(self.params['embedding'])
        else:
            self.embedding_block = None
        
        # Build trunk blocks
        self.trunk_blocks = nn.ModuleList()
        for block_params in self.params.get('trunk', []):
            self.trunk_blocks.append(self.build_block(block_params))
        
        # Build head blocks
        self.head_blocks = nn.ModuleDict()
        for head_name, head_config in self.params.get('heads', {}).items():
            if isinstance(head_config, list):
                # List of blocks for this head
                head_layers = nn.ModuleList()
                for block_params in head_config:
                    head_layers.append(self.build_block(block_params))
                self.head_blocks[head_name] = head_layers
            else:
                # Single block for this head
                self.head_blocks[head_name] = nn.ModuleList([self.build_block(head_config)])
        
        if self.params.get('verbose', False):
            print(f"Built model with {len(self.trunk_blocks)} trunk blocks and {len(self.head_blocks)} heads")
    
    def build_block(self, block_params):
        """Build a single block from parameters.
        
        Args:
            block_params: Dictionary with 'name' and block-specific parameters
        
        Returns:
            PyTorch module for the block
        """
        if isinstance(block_params, dict):
            block_params = block_params.copy()
            block_name = block_params.pop('name')
        else:
            raise ValueError(f"Block params must be dict, got {type(block_params)}")
        
        # Add global defaults if not specified
        global_vars = ['dropout', 'activation', 'l2_scale', 'kernel_size']
        for gv in global_vars:
            if gv in self.params and gv not in block_params:
                block_params[gv] = self.params[gv]
        
        # Get block function
        if block_name not in blocks.name_func:
            raise ValueError(f"Unknown block type: {block_name}. Available: {list(blocks.name_func.keys())}")
        
        block_func = blocks.name_func[block_name]
        
        # Build and return block
        try:
            return block_func(**block_params)
        except TypeError as e:
            raise TypeError(f"Error building block '{block_name}': {e}\nParams: {block_params}")
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the dynamically built network.
        
        Args:
            x: Input tensor with shape:
               - (batch, seq_length) for token encoding {0, 1, 2, 3}
               - (batch, seq_length, 8) for diploid_onehot encoding
            return_embeddings: If True, also return intermediate embeddings
            
        Returns:
            Dictionary containing:
                - '<head_name>': Output for each head
                - 'embeddings': (batch, embed_dim) if return_embeddings=True
        """
        # Create mask for non-missing values
        # For token encoding: mask out token 3
        # For diploid_onehot: mask out all-zero vectors
        if x.ndim == 2:
            # Token encoding: (batch, seq_length)
            mask = (x != 3)
        elif x.ndim == 3:
            # Diploid one-hot encoding: (batch, seq_length, 8)
            # Missing is represented as all zeros
            mask = (x.sum(dim=-1) > 0)  # (batch, seq_length)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Embedding
        if self.embedding_block is not None:
            current = self.embedding_block(x)
        else:
            # If no embedding specified, assume x is already embedded
            current = x
        
        # Trunk blocks
        for block in self.trunk_blocks:
            # Check if block accepts mask argument
            if self._block_accepts_mask(block):
                result = block(current, mask=mask)
                # Handle blocks that return (output, mask) tuple
                if isinstance(result, tuple):
                    current, mask = result
                else:
                    current = result
            else:
                current = block(current)
        
        # Save trunk output for embeddings
        trunk_output = current
        
        # Head blocks
        outputs = {}
        for head_name, head_layers in self.head_blocks.items():
            head_current = trunk_output
            for layer in head_layers:
                if self._block_accepts_mask(layer):
                    head_current = layer(head_current, mask=mask)
                else:
                    head_current = layer(head_current)
            outputs[head_name] = head_current
        
        if return_embeddings:
            outputs['embeddings'] = trunk_output
        
        return outputs
    
    def _block_accepts_mask(self, block):
        """Check if a block accepts a mask argument."""
        # Check if forward method accepts mask parameter
        import inspect
        sig = inspect.signature(block.forward)
        return 'mask' in sig.parameters
    
    def get_task_predictions(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        Get predictions for a specific task.
        
        Args:
            x: (batch, seq_length) SNP tensor
            task_name: Name of the task
            
        Returns:
            Predictions for the specified task
        """
        outputs = self.forward(x)
        
        if task_name in self.regression_tasks:
            idx = self.regression_tasks.index(task_name)
            if 'regression' in outputs:
                return outputs['regression'][:, idx]
            else:
                raise ValueError(f"No regression head found in model outputs: {outputs.keys()}")
        elif task_name in self.classification_tasks:
            idx = self.classification_tasks.index(task_name)
            if 'classification' in outputs:
                logits = outputs['classification'][:, idx]
                return torch.sigmoid(logits)  # For binary classification
            else:
                raise ValueError(f"No classification head found in model outputs: {outputs.keys()}")
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def count_parameters(self) -> Dict[str, int]:
        """Count the number of parameters in each component."""
        counts = {}
        
        if self.embedding_block is not None:
            counts['embedding'] = sum(p.numel() for p in self.embedding_block.parameters())
        
        counts['trunk'] = sum(p.numel() for p in self.trunk_blocks.parameters())
        
        for head_name, head_layers in self.head_blocks.items():
            counts[f'head_{head_name}'] = sum(p.numel() for p in head_layers.parameters())
        
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return counts


def create_model_from_config(
    config: dict,
    seq_length: int,
    regression_tasks: Optional[List[str]] = None,
    classification_tasks: Optional[List[str]] = None
) -> SNPNeuralNetwork:
    """
    Create a SNPNeuralNetwork from a configuration dictionary.
    
    Automatically detects and fills in task dimensions based on provided task lists.
    
    Args:
        config: Configuration dictionary with model parameters
        seq_length: Sequence length (number of SNPs)
        regression_tasks: List of regression task column names
        classification_tasks: List of classification task column names
        
    Returns:
        Initialized SNPNeuralNetwork
    """
    import copy
    
    # Deep copy to avoid modifying original config
    model_params = copy.deepcopy(config.get('model', {}))
    
    # Set sequence length and tasks
    model_params['seq_length'] = seq_length
    model_params['regression_tasks'] = regression_tasks or []
    model_params['classification_tasks'] = classification_tasks or []
    
    # Auto-update head dimensions based on detected tasks
    if 'heads' in model_params:
        # Update regression head
        if 'regression' in model_params['heads'] and regression_tasks:
            head_blocks = model_params['heads']['regression']
            if isinstance(head_blocks, list):
                for block in head_blocks:
                    if isinstance(block, dict) and block.get('name') == 'regression_head':
                        block['num_targets'] = len(regression_tasks)
            elif isinstance(head_blocks, dict) and head_blocks.get('name') == 'regression_head':
                head_blocks['num_targets'] = len(regression_tasks)
        
        # Update classification head
        if 'classification' in model_params['heads'] and classification_tasks:
            head_blocks = model_params['heads']['classification']
            if isinstance(head_blocks, list):
                for block in head_blocks:
                    if isinstance(block, dict) and block.get('name') == 'classification_head':
                        block['num_tasks'] = len(classification_tasks)
            elif isinstance(head_blocks, dict) and head_blocks.get('name') == 'classification_head':
                head_blocks['num_tasks'] = len(classification_tasks)
    
    return SNPNeuralNetwork(model_params)
