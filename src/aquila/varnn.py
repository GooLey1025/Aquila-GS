"""
Dynamic neural network architecture for genomic variant-based prediction.
Builds models from YAML configuration using composable blocks.
Supports single-branch and multi-branch architectures for SNP/INDEL/SV variants.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from . import blocks


class VariantsNeuralNetwork(nn.Module):
    """
    Multi-task neural network for genomic prediction with dynamic architecture.
    
    The architecture is built from YAML configuration specifying:
        1. Embedder layer (variant encoding)
        2. Trunk blocks (shared feature extraction)
        3. Head blocks (task-specific predictions)
    
    Supports both single-input (tensor) and multi-branch (dict) architectures.
    """
    
    def __init__(self, params: dict):
        """Initialize model from configuration parameters.
        
        Args:
            params: Configuration dictionary with:
                - seq_length: Sequence length
                - embedder: Embedder block config
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
    
    def build_model(self):
        """Build model dynamically from trunk and head configs."""
        # Build embedder if specified (can be a single block or list of blocks)
        if 'embedder' in self.params:
            embedder_config = self.params['embedder']
            if isinstance(embedder_config, list):
                # Multiple embedder blocks
                self.embedder = nn.ModuleList()
                for block_params in embedder_config:
                    self.embedder.append(self.build_block(block_params))
            else:
                # Single embedder block
                self.embedder = self.build_block(embedder_config)
        else:
            self.embedder = None
        
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
        
        # Embedder (can be single block or ModuleList)
        if self.embedder is not None:
            if isinstance(self.embedder, nn.ModuleList):
                # Multiple embedder blocks: apply sequentially
                current = x
                for embedder_block in self.embedder:
                    result = embedder_block(current)
                    # Handle blocks that return (output, mask) tuple
                    if isinstance(result, tuple):
                        current, mask = result
                    else:
                        current = result
            else:
                # Single embedder block
                result = self.embedder(x)
                # Handle blocks that return (output, mask) tuple
                if isinstance(result, tuple):
                    current, mask = result
                else:
                    current = result
        else:
            # If no embedder specified, assume x is already embedded
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
        
        if self.embedder is not None:
            if isinstance(self.embedder, nn.ModuleList):
                counts['embedder'] = sum(p.numel() for p in self.embedder.parameters())
            else:
                counts['embedder'] = sum(p.numel() for p in self.embedder.parameters())
        
        counts['trunk'] = sum(p.numel() for p in self.trunk_blocks.parameters())
        
        for head_name, head_layers in self.head_blocks.items():
            counts[f'head_{head_name}'] = sum(p.numel() for p in head_layers.parameters())
        
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return counts


class MultiBranchNeuralNetwork(VariantsNeuralNetwork):
    """
    Multi-branch neural network for genomic prediction with variant-type-specific branches.
    
    Architecture:
        - Multiple input branches (SNP, INDEL, SV), each with its own embedder and trunk
        - Fusion layer(s) to combine branch outputs
        - Shared trunk for post-fusion processing
        - Task-specific heads for predictions
    
    Config structure:
        branches:
            snp: {embedder: {...}, trunk: [...]}
            indel: {embedder: {...}, trunk: [...]}
            sv: {embedder: {...}, trunk: [...]}
        fusion: [{name: gated_fusion, ...}, {name: cross_attention_fusion, ...}]
        shared_trunk: [...]
        heads: {regression: [...], classification: [...]}
    """
    
    def __init__(self, params: dict):
        """Initialize multi-branch model from configuration.
        
        Args:
            params: Configuration dictionary with branches, fusion, shared_trunk, heads
        """
        # Don't call super().__init__() yet, we'll build differently
        nn.Module.__init__(self)
        
        self.params = params
        self.seq_length = params.get('seq_length', {})  # Dict for multi-branch
        
        # Task configuration
        self.regression_tasks = params.get('regression_tasks', [])
        self.classification_tasks = params.get('classification_tasks', [])
        self.num_regression_tasks = len(self.regression_tasks)
        self.num_classification_tasks = len(self.classification_tasks)
        
        # Build multi-branch model
        self.build_multi_branch_model()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def build_multi_branch_model(self):
        """Build multi-branch architecture from config."""
        branches_config = self.params.get('branches', {})
        
        # Build each branch
        self.branch_embedders = nn.ModuleDict()
        self.branch_trunks = nn.ModuleDict()
        
        for branch_name, branch_config in branches_config.items():
            # Build embedder for this branch
            if 'embedder' in branch_config:
                self.branch_embedders[branch_name] = self.build_block(branch_config['embedder'])
            
            # Build trunk for this branch
            trunk_blocks = nn.ModuleList()
            for block_params in branch_config.get('trunk', []):
                trunk_blocks.append(self.build_block(block_params))
            self.branch_trunks[branch_name] = trunk_blocks
        
        # Build fusion blocks
        self.fusion_blocks = nn.ModuleList()
        for fusion_params in self.params.get('fusion', []):
            self.fusion_blocks.append(self.build_block(fusion_params))
        
        # Build shared trunk (after fusion)
        self.shared_trunk_blocks = nn.ModuleList()
        for block_params in self.params.get('shared_trunk', []):
            self.shared_trunk_blocks.append(self.build_block(block_params))
        
        # Build head blocks
        self.head_blocks = nn.ModuleDict()
        for head_name, head_config in self.params.get('heads', {}).items():
            if isinstance(head_config, list):
                head_layers = nn.ModuleList()
                for block_params in head_config:
                    head_layers.append(self.build_block(block_params))
                self.head_blocks[head_name] = head_layers
            else:
                self.head_blocks[head_name] = nn.ModuleList([self.build_block(head_config)])
        
        if self.params.get('verbose', False):
            print(f"Built multi-branch model with {len(self.branch_trunks)} branches, "
                  f"{len(self.fusion_blocks)} fusion blocks, "
                  f"{len(self.shared_trunk_blocks)} shared trunk blocks, "
                  f"and {len(self.head_blocks)} heads")
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-branch network.
        
        Args:
            x: Dict of input tensors: {'snp': tensor, 'indel': tensor, 'sv': tensor}
               Each tensor has shape (batch, seq_length, 8) for diploid_onehot
            return_embeddings: If True, also return intermediate embeddings
            
        Returns:
            Dictionary containing head outputs
        """
        if not isinstance(x, dict):
            raise ValueError(f"MultiBranchNeuralNetwork expects dict input, got {type(x)}")
        
        # Process each branch
        branch_outputs = {}
        for branch_name, branch_input in x.items():
            if branch_name not in self.branch_trunks:
                continue
            
            # Embedder
            if branch_name in self.branch_embedders:
                current = self.branch_embedders[branch_name](branch_input)
            else:
                current = branch_input
            
            # Branch trunk
            for block in self.branch_trunks[branch_name]:
                current = block(current)
            
            branch_outputs[branch_name] = current
        
        # Apply fusion
        # Convert dict to list in consistent order
        branch_list = [branch_outputs[name] for name in sorted(branch_outputs.keys())]
        
        fused = branch_list[0]  # Start with first branch
        for fusion_block in self.fusion_blocks:
            if isinstance(fusion_block, blocks.GatedFusionBlock):
                # Gated fusion takes list of all branches
                fused = fusion_block(branch_list)
            elif isinstance(fusion_block, blocks.CrossAttentionFusionBlock):
                # Cross-attention: query = fused, kv = all branches
                fused = fusion_block(fused.unsqueeze(1) if fused.dim() == 2 else fused, branch_list)
            else:
                # Generic fusion block
                fused = fusion_block(fused)
        
        # Shared trunk
        for block in self.shared_trunk_blocks:
            fused = block(fused)
        
        trunk_output = fused
        
        # Head blocks
        outputs = {}
        for head_name, head_layers in self.head_blocks.items():
            head_current = trunk_output
            for layer in head_layers:
                head_current = layer(head_current)
            outputs[head_name] = head_current
        
        if return_embeddings:
            outputs['embeddings'] = trunk_output
        
        return outputs


def create_model_from_config(
    config: dict,
    seq_length: int,
    regression_tasks: Optional[List[str]] = None,
    classification_tasks: Optional[List[str]] = None
) -> VariantsNeuralNetwork:
    """
    Create a VariantsNeuralNetwork from a configuration dictionary.
    
    Automatically detects and fills in task dimensions based on provided task lists.
    Detects architecture_type to create single-branch or multi-branch models.
    
    Args:
        config: Configuration dictionary with model parameters
        seq_length: Sequence length (number of variants) or dict for multi-branch
        regression_tasks: List of regression task column names
        classification_tasks: List of classification task column names
        
    Returns:
        Initialized VariantsNeuralNetwork or MultiBranchNeuralNetwork
    """
    import copy
    
    # Deep copy to avoid modifying original config
    model_params = copy.deepcopy(config.get('model', {}))
    
    # Check for multi-branch architecture
    architecture_type = model_params.get('architecture_type', 'single')
    
    # Set tasks
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
    
    if architecture_type == 'multi_branch':
        # Multi-branch architecture
        model_params['seq_length'] = seq_length  # Can be dict for multi-branch
        return MultiBranchNeuralNetwork(model_params)
    else:
        # Single-branch architecture
        model_params['seq_length'] = seq_length
        return VariantsNeuralNetwork(model_params)
