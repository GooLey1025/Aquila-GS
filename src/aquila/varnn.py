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
        self.embedder_save_as: list[str | None] = []
        if 'embedder' in self.params:
            embedder_config = self.params['embedder']
            if isinstance(embedder_config, list):
                self.embedder = nn.ModuleList()
                for block_params in embedder_config:
                    cleaned, save_as = self._split_save_as(block_params)
                    self.embedder_save_as.append(save_as)
                    self.embedder.append(self.build_block(cleaned))
            else:
                cleaned, save_as = self._split_save_as(embedder_config)
                self.embedder_save_as.append(save_as)
                self.embedder = self.build_block(cleaned)
        else:
            self.embedder = None
        
        # Build trunk blocks
        self.trunk_blocks = nn.ModuleList()
        self.trunk_save_as: list[str | None] = []
        for block_params in self.params.get('trunk', []):
            cleaned, save_as = self._split_save_as(block_params)
            self.trunk_save_as.append(save_as)
            self.trunk_blocks.append(self.build_block(cleaned))
        
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
    
    @staticmethod
    def _split_save_as(block_params):
        if not isinstance(block_params, dict):
            raise ValueError(f"Block params must be dict, got {type(block_params)}")
        cleaned = block_params.copy()
        save_as = cleaned.pop('save_as', None)
        return cleaned, save_as

    def _run_trunk_block(self, block, current, mask, saved):
        from aquila.blocks import SkipFusePool, SkipFuseSeq

        if isinstance(block, (SkipFuseSeq, SkipFusePool)):
            if block.source not in saved:
                raise KeyError(
                    f"SkipFuse source '{block.source}' was not saved. "
                    f"Available: {sorted(saved)}"
                )
            fused = block(current, saved[block.source], mask=mask)
            return fused, mask
        if self._block_accepts_mask(block):
            result = block(current, mask=mask)
            if isinstance(result, tuple):
                return result
            return result, mask
        return block(current), mask

    def build_block(self, block_params):
        """Build a single block from parameters.
        
        Args:
            block_params: Dictionary with 'name' and block-specific parameters
        
        Returns:
            PyTorch module for the block
        """
        if isinstance(block_params, dict):
            block_params = block_params.copy()
            block_params.pop('save_as', None)
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
               - (batch, seq_length, 3) for classic onehot (REF/HET/ALT)
            return_embeddings: If True, also return intermediate embeddings
            
        Returns:
            Dictionary containing:
                - '<head_name>': Output for each head
                - 'embeddings': (batch, embed_dim) if return_embeddings=True
        """
        # Create mask for non-missing values
        # For token encoding: mask out token 3
        # For float one-hot encodings: mask out all-zero vectors (missing)
        if x.ndim == 2:
            # Token encoding: (batch, seq_length)
            mask = (x != 3)
        elif x.ndim == 3:
            # Diploid one-hot (8) or classic onehot (3): missing = all zeros
            mask = (x.sum(dim=-1) > 0)  # (batch, seq_length)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        saved: Dict[str, torch.Tensor] = {}
        # Embedder (can be single block or ModuleList)
        if self.embedder is not None:
            if isinstance(self.embedder, nn.ModuleList):
                current = x
                for embedder_block, save_as in zip(self.embedder, self.embedder_save_as):
                    result = embedder_block(current)
                    if isinstance(result, tuple):
                        current, mask = result
                    else:
                        current = result
                    if save_as:
                        saved[save_as] = current
            else:
                result = self.embedder(x)
                if isinstance(result, tuple):
                    current, mask = result
                else:
                    current = result
                for save_as in self.embedder_save_as:
                    if save_as:
                        saved[save_as] = current
        else:
            current = x
        
        # Trunk blocks
        trunk_saves = getattr(self, "trunk_save_as", [None] * len(self.trunk_blocks))
        for block, save_as in zip(self.trunk_blocks, trunk_saves):
            current, mask = self._run_trunk_block(block, current, mask, saved)
            if save_as:
                saved[save_as] = current
        
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
        
    def build_multi_branch_model(self):
        """Build multi-branch architecture from config."""
        branches_config = self.params.get('branches', {})
        
        # Build each branch
        self.branch_embedders = nn.ModuleDict()
        self.branch_trunks = nn.ModuleDict()
        
        for branch_name, branch_config in branches_config.items():
            # Build embedder for this branch (support both 'embedder' and 'embedding' keys)
            embedder_config = branch_config.get('embedder') or branch_config.get('embedding')
            if embedder_config:
                # Support both single dict and list of blocks (for stacked embedders)
                if isinstance(embedder_config, list):
                    embedder_blocks = nn.ModuleList()
                    for block_params in embedder_config:
                        embedder_blocks.append(self.build_block(block_params))
                    self.branch_embedders[branch_name] = embedder_blocks
                else:
                    self.branch_embedders[branch_name] = self.build_block(embedder_config)
            
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
               Each tensor has shape (batch, seq_length, 8) for diploid_onehot (or 3 for onehot if used)
            return_embeddings: If True, also return intermediate embeddings
            
        Returns:
            Dictionary containing head outputs
        """
        if not isinstance(x, dict):
            raise ValueError(f"MultiBranchNeuralNetwork expects dict input, got {type(x)}")
        
        # Normalize input keys to lowercase to match model config keys
        # Data loader may use uppercase keys like 'SNP', 'INDEL', 'SV'
        # Model config uses lowercase like 'snp', 'indel', 'sv'
        x_normalized = {k.lower(): v for k, v in x.items()}
        
        # Process each branch
        branch_outputs = {}
        for branch_name, branch_input in x_normalized.items():
            if branch_name not in self.branch_trunks:
                continue
            
            # Embedder
            if branch_name in self.branch_embedders:
                embedder = self.branch_embedders[branch_name]
                if isinstance(embedder, nn.ModuleList):
                    current = branch_input
                    for embedder_block in embedder:
                        result = embedder_block(current)
                        # Handle blocks that return (x, mask) tuple
                        if isinstance(result, tuple):
                            current = result[0]  # Use only the tensor, ignore mask
                        else:
                            current = result
                else:
                    result = embedder(branch_input)
                    # Handle blocks that return (x, mask) tuple
                    if isinstance(result, tuple):
                        current = result[0]
                    else:
                        current = result
            else:
                current = branch_input
            
            # Branch trunk
            for block in self.branch_trunks[branch_name]:
                result = block(current)
                # Handle blocks that return (x, mask) tuple
                if isinstance(result, tuple):
                    current = result[0]
                else:
                    current = result
            
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
                # Cross-attention: pass all branches, uses which_branch_as_query to select
                fused = fusion_block(*branch_list)
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
    train_config = config.get('train', {})

    # Check for multi-branch architecture
    # First check if model has architecture_type, then check if train.branches exists
    architecture_type = model_params.get('architecture_type', 'single')

    # Handle multi-branch config that uses train.branches instead of model.branches
    # This supports configs like aquila-vars.hpo.yaml where branches are in train section
    if architecture_type == 'multi_branch' or 'branches' in train_config:
        # If branches are in train config but not in model config, use train.branches
        if 'branches' not in model_params and 'branches' in train_config:
            model_params['branches'] = train_config['branches']

        # Also extract fusion, shared_trunk, heads from train config if present
        # These are model-related configs that some HPO configs put under train section
        if 'fusion' not in model_params and 'fusion' in train_config:
            model_params['fusion'] = train_config['fusion']
        if 'shared_trunk' not in model_params and 'shared_trunk' in train_config:
            model_params['shared_trunk'] = train_config['shared_trunk']
        if 'heads' not in model_params and 'heads' in train_config:
            model_params['heads'] = train_config['heads']

        # Set architecture_type to multi_branch if using train.branches
        if 'branches' in model_params:
            architecture_type = 'multi_branch'

    # Set tasks
    model_params['regression_tasks'] = regression_tasks or []
    model_params['classification_tasks'] = classification_tasks or []

    # Auto-update head dimensions based on detected tasks
    _MULTI_TASK_REGRESSION_HEADS = {
        'regression_head',
        'per_trait_regression_head',
        'shared_stem_private_head',
        'family_grouped_regression_head',
        'shared_stem_family_head',
        'mmoe_regression_head',
        'film_regression_head',
        'trait_query_regression_head',
    }
    _FAMILY_REGRESSION_HEADS = {
        'family_grouped_regression_head',
        'shared_stem_family_head',
    }
    if 'heads' in model_params:
        # Update regression head
        if 'regression' in model_params['heads'] and regression_tasks:
            head_blocks = model_params['heads']['regression']
            if isinstance(head_blocks, list):
                for block in head_blocks:
                    if isinstance(block, dict) and block.get('name') in _MULTI_TASK_REGRESSION_HEADS:
                        block['num_targets'] = len(regression_tasks)
                        if block.get('name') in _FAMILY_REGRESSION_HEADS:
                            block['task_names'] = list(regression_tasks)
            elif isinstance(head_blocks, dict) and head_blocks.get('name') in _MULTI_TASK_REGRESSION_HEADS:
                head_blocks['num_targets'] = len(regression_tasks)
                if head_blocks.get('name') in _FAMILY_REGRESSION_HEADS:
                    head_blocks['task_names'] = list(regression_tasks)

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
