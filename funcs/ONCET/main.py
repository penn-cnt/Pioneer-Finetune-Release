"""
main.py

Main pipeline for seizure detection project.
Orchestrates data loading, training, cross-validation, and evaluation.
"""

import argparse
import sys
from pathlib import Path
import json
import torch
from typing import List, Optional

from config import ExperimentConfig, create_default_config, create_cv_config, create_final_training_config
from data_utils import SeizureDataset, create_dataloaders
from model import LightweightSeizureDetector, count_parameters
from train import Trainer, create_optimizer, create_scheduler, create_criterion
from cross_validation import run_hyperparameter_optimization
from evaluate import evaluate_model_from_checkpoint
from metrics import print_metrics


def split_patients(
    dataset: SeizureDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple:
    """
    Split patients into train/val/test sets.
    
    Args:
        dataset: SeizureDataset
        train_ratio: Ratio of patients for training
        val_ratio: Ratio of patients for validation
        test_ratio: Ratio of patients for testing
        random_state: Random seed
        
    Returns:
        (train_patient_ids, val_patient_ids, test_patient_ids)
    """
    import numpy as np
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(random_state)
    
    # Get all patient IDs
    patient_ids = dataset.get_patient_ids()
    n_patients = len(patient_ids)
    
    # Shuffle
    shuffled_patients = np.random.permutation(patient_ids)
    
    # Split
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = shuffled_patients[:n_train].tolist()
    val_patients = shuffled_patients[n_train:n_train + n_val].tolist()
    test_patients = shuffled_patients[n_train + n_val:].tolist()
    
    print(f"\nPatient split:")
    print(f"  Train: {len(train_patients)} patients ({len(train_patients)/n_patients*100:.1f}%)")
    print(f"  Val:   {len(val_patients)} patients ({len(val_patients)/n_patients*100:.1f}%)")
    print(f"  Test:  {len(test_patients)} patients ({len(test_patients)/n_patients*100:.1f}%)")
    
    return train_patients, val_patients, test_patients


def train_model(config: ExperimentConfig, save_dir: Optional[str] = None):
    """
    Train a single model with given configuration.
    
    Args:
        config: Experiment configuration
        save_dir: Directory to save results (uses config.output_dir if None)
    """
    if save_dir is None:
        save_dir = Path(config.output_dir) / 'training'
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    config.print_config()
    
    # Save config
    config.save(save_dir / 'config.yaml')
    
    # Load dataset and split patients if needed
    dataset = SeizureDataset(config.data.h5_path, combine_onset_spread=True)
    
    if config.data.train_patient_ids is None:
        print("\nNo patient split provided. Creating automatic split...")
        train_patients, val_patients, test_patients = split_patients(
            dataset,
            random_state=config.seed
        )
        
        # Update config
        config.data.train_patient_ids = train_patients
        config.data.val_patient_ids = val_patients
        config.data.test_patient_ids = test_patients
        
        # Save updated config
        config.save(save_dir / 'config.yaml')
        
        # Save patient splits
        with open(save_dir / 'patient_splits.json', 'w') as f:
            json.dump({
                'train': train_patients,
                'val': val_patients,
                'test': test_patients
            }, f, indent=4)
    else:
        train_patients = config.data.train_patient_ids
        val_patients = config.data.val_patient_ids
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        config.data.h5_path,
        train_patients,
        val_patients,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        normalize_per_sample=config.data.normalize_per_sample
    )
    
    # Create model
    print("\nCreating model...")
    model = LightweightSeizureDetector(
        num_classes=config.model.num_classes,
        base_filters=config.model.base_filters,
        num_blocks=config.model.num_blocks,
        kernel_size=config.model.kernel_size,
        dilations=config.model.dilations,
        dropout_stem=config.model.dropout_stem,
        dropout_blocks=config.model.dropout_blocks,
        dropout_head=config.model.dropout_head
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=config.training.scheduler,
        num_epochs=config.training.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.training.pct_start,
        max_lr=config.training.max_lr or config.training.learning_rate
    )
    
    # Create criterion
    if config.training.use_class_weights:
        train_dist = train_loader.dataset.get_class_distribution()
        n_samples = sum(train_dist.values())
        n_classes = len(train_dist)
        class_weights = torch.tensor([
            n_samples / (n_classes * train_dist[i]) 
            for i in range(n_classes)
        ], dtype=torch.float32)
        class_weights = class_weights.to(config.training.device)
    else:
        class_weights = None
    
    criterion = create_criterion(
        config.training.criterion,
        class_weights=class_weights,
        label_smoothing=config.training.label_smoothing,
        device=config.training.device
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.training.device,
        grad_clip=config.training.grad_clip,
        early_stopping_patience=config.training.early_stopping_patience
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        num_epochs=config.training.num_epochs,
        save_dir=save_dir / 'checkpoints',
        verbose=True
    )
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_json = {
            key: [float(v) for v in values]
            for key, values in history.items()
        }
        json.dump(history_json, f, indent=4)
    
    print(f"\nTraining complete! Results saved to {save_dir}/")
    
    return trainer, history


def run_cross_validation(config: ExperimentConfig):
    """
    Run cross-validation with hyperparameter optimization.
    
    Args:
        config: Experiment configuration
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION WITH HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    config.print_config()
    
    save_dir = Path(config.output_dir) / 'cross_validation'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(save_dir / 'config.yaml')
    
    # Prepare config for optimization
    cv_config = {
        'max_epochs': config.cv.max_epochs_per_trial,
        'n_folds': config.cv.n_folds,
        'num_workers': config.data.num_workers,
        'device': config.training.device,
        'grad_clip': config.training.grad_clip,
        'early_stopping_patience': config.cv.early_stopping_patience,
        'use_class_weights': config.training.use_class_weights,
        'random_state': config.cv.random_state,
        'output_dir': str(save_dir)
    }
    
    # Run optimization
    study = run_hyperparameter_optimization(
        h5_path=config.data.h5_path,
        config=cv_config,
        n_trials=config.cv.n_trials,
        study_name=config.cv.study_name
    )
    
    print(f"\nCross-validation complete! Results saved to {save_dir}/")
    
    return study


def evaluate_test_set(
    config: ExperimentConfig,
    checkpoint_path: str,
    test_patient_ids: Optional[List[str]] = None
):
    """
    Evaluate trained model on test set.
    
    Args:
        config: Experiment configuration
        checkpoint_path: Path to model checkpoint
        test_patient_ids: List of test patient IDs (uses config if None)
    """
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    save_dir = Path(config.output_dir) / 'test_evaluation'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test patient IDs
    if test_patient_ids is None:
        if config.data.test_patient_ids is None:
            # Load from patient splits file
            splits_path = Path(config.output_dir) / 'training' / 'patient_splits.json'
            if splits_path.exists():
                with open(splits_path, 'r') as f:
                    splits = json.load(f)
                test_patient_ids = splits['test']
            else:
                raise ValueError("No test patient IDs provided and no patient splits file found")
        else:
            test_patient_ids = config.data.test_patient_ids
    
    # Create model config dict
    model_config = {
        'num_classes': config.model.num_classes,
        'base_filters': config.model.base_filters,
        'num_blocks': config.model.num_blocks,
        'kernel_size': config.model.kernel_size,
        'dilations': config.model.dilations,
        'dropout_blocks': config.model.dropout_blocks,
        'dropout_head': config.model.dropout_head,
        'normalize_per_sample': config.data.normalize_per_sample
    }
    
    # Evaluate
    metrics = evaluate_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        h5_path=config.data.h5_path,
        test_patient_ids=test_patient_ids,
        config=model_config,
        save_dir=str(save_dir)
    )
    
    print(f"\nTest evaluation complete! Results saved to {save_dir}/")
    
    return metrics


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Seizure Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python main.py --mode train
  
  # Train with custom config
  python main.py --mode train --config configs/my_config.yaml
  
  # Run cross-validation
  python main.py --mode cv --config configs/cv_config.yaml
  
  # Evaluate on test set
  python main.py --mode evaluate --checkpoint results/training/checkpoints/best_model.pth
  
  # Full pipeline: CV -> train with best params -> evaluate
  python main.py --mode pipeline --config configs/cv_config.yaml
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'cv', 'evaluate', 'pipeline'],
        help='Mode to run: train, cv (cross-validation), evaluate, or pipeline (full workflow)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (.yaml or .json)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (for evaluation mode)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = ExperimentConfig.load(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        if args.mode == 'cv':
            config = create_cv_config()
            print("Using default cross-validation configuration")
        else:
            config = create_default_config()
            print("Using default configuration")
    
    # Override output dir if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run requested mode
    if args.mode == 'train':

        config = ExperimentConfig.load(args.config)
        trainer, history = train_model(config)
        print("\n✓ Training complete!")
        
    elif args.mode == 'cv':
        study = run_cross_validation(config)
        print("\n✓ Cross-validation complete!")
        print(f"\nTo train final model with best hyperparameters:")
        print(f"  1. Load best hyperparameters from: {config.output_dir}/cross_validation/best_hyperparameters.json")
        print(f"  2. Create config with: create_final_training_config(best_params)")
        print(f"  3. Run: python main.py --mode train --config <new_config>")
        
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            print("Error: --checkpoint required for evaluate mode")
            sys.exit(1)
        
        metrics = evaluate_test_set(config, args.checkpoint)
        print("\n✓ Evaluation complete!")
        
    elif args.mode == 'pipeline':
        print("\n" + "="*60)
        print("FULL PIPELINE: CV -> TRAIN -> EVALUATE")
        print("="*60)
        
        # Step 1: Cross-validation
        print("\n[STEP 1/3] Running cross-validation...")
        study = run_cross_validation(config)
        
        # Step 2: Load best hyperparameters and train final model
        print("\n[STEP 2/3] Training final model with best hyperparameters...")
        best_params_path = Path(config.output_dir) / 'cross_validation' / 'best_hyperparameters.json'
        with open(best_params_path, 'r') as f:
            best_params_data = json.load(f)
        
        best_hyperparameters = best_params_data['hyperparameters']
        
        # Create final training config
        final_config = create_final_training_config(best_hyperparameters)
        final_config.output_dir = str(Path(config.output_dir) / 'final_training')
        
        # Use same patient splits from CV
        final_config.data.h5_path = config.data.h5_path
        
        # Train final model
        trainer, history = train_model(final_config)
        
        # Step 3: Evaluate on test set
        print("\n[STEP 3/3] Evaluating on test set...")
        checkpoint_path = Path(final_config.output_dir) / 'training' / 'checkpoints' / 'best_model.pth'
        metrics = evaluate_test_set(final_config, str(checkpoint_path))
        
        print("\n" + "="*60)
        print("✓ FULL PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {config.output_dir}/")
        print(f"  - Cross-validation:  {config.output_dir}/cross_validation/")
        print(f"  - Final training:    {config.output_dir}/final_training/")
        print(f"  - Test evaluation:   {config.output_dir}/final_training/test_evaluation/")


if __name__ == "__main__":
    main()