"""
Advanced Neural Network Models
Includes TCN, GNN, and Transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalConvolutionNetwork(nn.Module):
    """
    Temporal Convolutional Network for expense trend analysis
    """
    
    def __init__(self, input_size: int, num_channels: list = [25, 25, 25],
                 kernel_size: int = 5, dropout: float = 0.3):
        """
        Initialize TCN
        
        Args:
            input_size: Number of input features
            num_channels: List of channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(TemporalConvolutionNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        layers = []
        dilation = 1
        
        # First layer
        layers.append(nn.Conv1d(input_size, num_channels[0], kernel_size,
                               dilation=dilation, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Intermediate layers
        for i in range(1, len(num_channels)):
            dilation *= 2
            layers.append(nn.Conv1d(num_channels[i-1], num_channels[i],
                                   kernel_size, dilation=dilation, padding='same'))
            layers.append(nn.BatchNorm1d(num_channels[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
        logger.info(f"TCN initialized with channels {num_channels}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor
        """
        # Transpose to (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # TCN layers
        x = self.tcn(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Output layer
        x = self.fc(x)
        
        return x


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for behavioral relationship modeling
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        """
        Initialize GNN
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of GNN layers
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.node_encoder = nn.Linear(input_size, hidden_size)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.message_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.update_layers = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        
        logger.info(f"GNN initialized with {num_layers} layers")
    
    def construct_adjacency_matrix(self, h: torch.Tensor) -> torch.Tensor:
        """
        Construct adjacency matrix based on node features
        
        Args:
            h: Node embeddings (batch_size, num_nodes, hidden_size)
            
        Returns:
            Adjacency matrix
        """
        batch_size, num_nodes, hidden_size = h.shape

        hi = h.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, hidden_size)
        hj = h.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_size)
        pair = torch.cat([hi, hj], dim=-1)
        logits = self.edge_mlp(pair).squeeze(-1)

        mask = torch.eye(num_nodes, device=h.device).unsqueeze(0)
        logits = logits.masked_fill(mask.bool(), float('-inf'))

        adj = torch.softmax(logits, dim=-1)
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        return adj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor
        """
        batch_size, num_nodes, _ = x.shape

        h = self.node_encoder(x)
        h = F.relu(h)
        h = self.dropout(h)

        adj = self.construct_adjacency_matrix(h)

        for msg_fc, upd in zip(self.message_layers, self.update_layers):
            m = torch.bmm(adj, h)
            m = msg_fc(m)
            m = F.relu(m)
            m = self.dropout(m)

            h_flat = h.reshape(batch_size * num_nodes, -1)
            m_flat = m.reshape(batch_size * num_nodes, -1)
            h_flat = upd(m_flat, h_flat)
            h = h_flat.reshape(batch_size, num_nodes, -1)

        graph_emb = torch.mean(h, dim=1)
        return self.output_layer(graph_emb)


class TransformerModel(nn.Module):
    """
    Transformer model for long-term dependency learning
    """
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, d_ff: int = 256, dropout: float = 0.1):
        """
        Initialize Transformer
        
        Args:
            input_size: Number of input features
            d_model: Model dimensionality
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Transformer initialized with {num_layers} layers, "
                   f"{nhead} heads, d_model={d_model}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor
        """
        # Embedding
        x = self.embedding(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class EnsembleStackingModel(nn.Module):
    """
    Ensemble stacking layer combining TCN, GNN, and Transformer
    """
    
    def __init__(self, input_size: int, use_tcn: bool = True,
                 use_gnn: bool = True, use_transformer: bool = True):
        """
        Initialize ensemble model
        
        Args:
            input_size: Number of input features
            use_tcn: Whether to use TCN
            use_gnn: Whether to use GNN
            use_transformer: Whether to use Transformer
        """
        super(EnsembleStackingModel, self).__init__()
        
        self.use_tcn = use_tcn
        self.use_gnn = use_gnn
        self.use_transformer = use_transformer
        
        # Initialize base models
        self.models = nn.ModuleDict()
        n_models = 0
        
        if use_tcn:
            self.models['tcn'] = TemporalConvolutionNetwork(input_size)
            n_models += 1
        
        if use_gnn:
            self.models['gnn'] = GraphNeuralNetwork(input_size)
            n_models += 1
        
        if use_transformer:
            self.models['transformer'] = TransformerModel(input_size)
            n_models += 1
        
        # Meta-learner for stacking
        self.meta_learner = nn.Sequential(
            nn.Linear(n_models, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        logger.info(f"Ensemble model initialized with {n_models} base models")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning ensemble output and base model outputs
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Ensemble output and individual model outputs
        """
        base_outputs = []
        
        if self.use_tcn:
            tcn_out = self.models['tcn'](x)
            base_outputs.append(tcn_out)
        
        if self.use_gnn:
            gnn_out = self.models['gnn'](x)
            base_outputs.append(gnn_out)
        
        if self.use_transformer:
            transformer_out = self.models['transformer'](x)
            base_outputs.append(transformer_out)
        
        # Stack outputs
        stacked = torch.cat(base_outputs, dim=1)
        
        # Meta-learner
        ensemble_output = self.meta_learner(stacked)
        
        return ensemble_output, torch.cat(base_outputs, dim=1)


def create_model(model_type: str, input_size: int, **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('tcn', 'gnn', 'transformer', 'ensemble')
        input_size: Number of input features
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    if model_type == 'tcn':
        return TemporalConvolutionNetwork(input_size, **kwargs)
    elif model_type == 'gnn':
        return GraphNeuralNetwork(input_size, **kwargs)
    elif model_type == 'transformer':
        return TransformerModel(input_size, **kwargs)
    elif model_type == 'ensemble':
        return EnsembleStackingModel(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
