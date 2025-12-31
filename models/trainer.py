"""
Model Training Module
Handles training, validation, and model management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Tuple, Dict, List
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive training pipeline for neural network models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size
            
        Returns:
            Training and validation data loaders
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Data loaders created. Train: {len(train_loader)}, "
                   f"Val: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(X_batch)
            output = out[0] if isinstance(out, tuple) else out
            
            # Loss computation
            loss = self.criterion(output, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                out = self.model(X_batch)
                output = out[0] if isinstance(out, tuple) else out
                
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
           epochs: int = 50, early_stopping_patience: int = 15) -> Dict[str, List]:
        """
        Train model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training completed!")
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            out = self.model(X_tensor)
            output = out[0] if isinstance(out, tuple) else out
            
            predictions = output.cpu().numpy()
        
        return predictions
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to load checkpoint from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Checkpoint loaded from {path}")
    
    def get_training_history(self) -> Dict[str, List]:
        """
        Get training history
        
        Returns:
            Training history dictionary
        """
        return self.training_history


def train_model_pipeline(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 50, batch_size: int = 32,
                        learning_rate: float = 0.001,
                        device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    """
    Complete training pipeline
    
    Args:
        model: Neural network model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        
    Returns:
        Trained model and training history
    """
    # Initialize trainer
    trainer = ModelTrainer(model, device=device, learning_rate=learning_rate)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(X_train, y_train, X_val, y_val,
                                                    batch_size=batch_size)
    
    # Train
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    
    return trainer.model, history
