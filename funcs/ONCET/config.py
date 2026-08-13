"""
config.py

Configuration management for seizure detection project.
Handles experiment configurations and hyperparameters.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import argparse

@dataclass
class DataConfig:
    """Data configuration."""
    h5_path: str = 'seizure_data.h5'
    train_patient_ids: Optional[list] = None
    val_patient_ids: Optional[list] = None
    test_patient_ids: Optional[list] = None
    batch_size: int = 128
    num_workers: int = 4
    combine_onset_spread: bool = True
    normalize_per_sample: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    num_classes: int = 2
    base_filters: int = 32
    num_blocks: int = 3
    kernel_size: int = 16
    dilations: list = field(default_factory=lambda: [2, 4, 8])
    dropout_stem: float = 0.1
    dropout_blocks: float = 0.2
    dropout_head: float = 0.3
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimizer
    optimizer: str = 'adamw'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = 'onecycle'  # 'onecycle', 'cosine', 'step', 'plateau', 'none'
    max_lr: Optional[float] = None  # For OneCycleLR
    pct_start: float = 0.3  # For OneCycleLR
    
    # Loss
    criterion: str = 'cross_entropy'
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    
    # Training dynamics
    num_epochs: int = 50
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    
    # Device
    device: str = 'cuda'
    mixed_precision: bool = True
    

@dataclass
class CrossValidationConfig:
    """Cross-validation configuration."""
    n_folds: int = 5
    n_trials: int = 50
    max_epochs_per_trial: int = 30
    early_stopping_patience: int = 5
    random_state: int = 42
    study_name: str = 'seizure_detection_cv'
    downsample_ratio: float = 0.2
    

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    find_optimal_threshold: bool = True
    threshold_metric: str = 'f2'  # 'f1', 'f2', 'youden'
    generate_plots: bool = True
    

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment metadata
    experiment_name: str = 'seizure_detection_experiment'
    output_dir: str = 'results'
    seed: int = 42
    
    def __post_init__(self):
        """Set random seeds after initialization."""
        self.set_seed(self.seed)
    
    @staticmethod
    def set_seed(seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'cv': asdict(self.cv),
            'evaluation': asdict(self.evaluation),
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'seed': self.seed
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            cv=CrossValidationConfig(**config_dict.get('cv', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            experiment_name=config_dict.get('experiment_name', 'seizure_detection_experiment'),
            output_dir=config_dict.get('output_dir', 'results'),
            seed=config_dict.get('seed', 42)
        )
    
    def update(self, **kwargs):
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Try to update sub-configs
                for sub_config_name in ['data', 'model', 'training', 'cv', 'evaluation']:
                    sub_config = getattr(self, sub_config_name)
                    if hasattr(sub_config, key):
                        setattr(sub_config, key, value)
                        break
    
    def print_config(self):
        """Pretty print configuration."""
        print("\n" + "="*60)
        print(f"EXPERIMENT CONFIGURATION: {self.experiment_name}")
        print("="*60)
        
        config_dict = self.to_dict()
        
        for section, params in config_dict.items():
            if isinstance(params, dict):
                print(f"\n{section.upper()}:")
                for key, value in params.items():
                    print(f"  {key:25s}: {value}")
            else:
                print(f"\n{section.upper()}: {params}")
        
        print("\n" + "="*60 + "\n")


def create_default_config() -> ExperimentConfig:
    """Create default configuration."""
    return ExperimentConfig(
        experiment_name='seizure_detection_default',
        output_dir='results/default',
        seed=42
    )


def create_cv_config() -> ExperimentConfig:
    """Create configuration for cross-validation experiments."""
    config = ExperimentConfig(
        experiment_name='seizure_detection_cv',
        output_dir='results/cross_validation',
        seed=42
    )
    
    # Override CV settings
    config.cv.n_trials = 30
    config.cv.n_folds = 3
    config.cv.max_epochs_per_trial = 50
    config.cv.downsample_ratio = 0.3
    
    # Shorter training for CV
    config.training.num_epochs = 50
    config.training.early_stopping_patience = 5

    # Data configs
    config.data.num_workers = 16
    config.data.h5_path = 'data/seizure_data.h5'

    # Test configs
    # config.data.h5_path = 'data/med_data.h5'
    # config.cv.downsample_ratio = None
    # config.cv.max_epochs_per_trial = 10
    # config.training.num_epochs = 10
    # config.training.early_stopping_patience = 2
    # config.cv.n_trials = 5
    return config


def create_final_training_config(best_hyperparameters: Dict) -> ExperimentConfig:
    """
    Create configuration for final training with best hyperparameters.
    
    Args:
        best_hyperparameters: Dictionary of best hyperparameters from CV
                             Can be nested under 'hyperparameters' key or flat
        
    Returns:
        ExperimentConfig with best hyperparameters
    """
    config = ExperimentConfig(
        experiment_name='seizure_detection_final',
        output_dir='results/final_training',
        seed=42
    )
    
    # Extract hyperparameters if nested
    params = best_hyperparameters.get('hyperparameters', best_hyperparameters)
    
    # Map hyperparameters to config attributes
    if 'learning_rate' in params:
        config.training.learning_rate = params['learning_rate']
    if 'weight_decay' in params:
        config.training.weight_decay = params['weight_decay']
    if 'base_filters' in params:
        config.model.base_filters = params['base_filters']
    if 'dropout_blocks' in params:
        config.model.dropout_blocks = params['dropout_blocks']
    if 'dropout_head' in params:
        config.model.dropout_head = params['dropout_head']
    if 'num_blocks' in params:
        config.model.num_blocks = params['num_blocks']
        # Update dilations to match num_blocks
        config.model.dilations = [2 ** (i + 1) for i in range(params['num_blocks'])]
    if 'batch_size' in params:
        config.data.batch_size = params['batch_size']
    if 'label_smoothing' in params:
        config.training.label_smoothing = params['label_smoothing']
    if 'scheduler' in params:
        config.training.scheduler = params['scheduler']
    
    # Use full training epochs for final model
    config.training.num_epochs = 200
    config.training.early_stopping_patience = 5
    config.data.num_workers = 16
    config.data.h5_path = 'data/seizure_data.h5'
    config.data.normalize_per_sample = False
    return config


# Example configuration templates
QUICK_TEST_CONFIG = {
    'experiment_name': 'quick_test',
    'output_dir': 'results/quick_test',
    'training': {
        'num_epochs': 5,
        'early_stopping_patience': 2
    },
    'cv': {
        'n_trials': 3,
        'n_folds': 2,
        'max_epochs_per_trial': 5
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--best_hyperparameters', type=str, default=None)
    args = parser.parse_args()
    
    # Create and save example configurations
    if args.mode == 'train':
        best_hyperparameters = json.load(open(args.best_hyperparameters))
        config = create_final_training_config(best_hyperparameters)
        config.save(f'{config.output_dir}/final_training_config.json')
    elif args.mode == 'cv':
        config = create_cv_config()
        config.save(f'{config.output_dir}/cv_config.json')
    elif args.mode == 'quick_test':
        config = ExperimentConfig.from_dict(QUICK_TEST_CONFIG)
        config.save(f'{config.output_dir}/quick_test_config.json')
    else:
        print("Invalid mode")
        sys.exit(1)
    
    config.print_config()