from functools import partial
import numpy as np
import torch
import torch.nn.utils
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# from tabpfn_client import TabPFNRegressor, init
from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator

from utils import *
import os
from scipy.stats import pearsonr
from datetime import datetime
import json


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(self, patience=3, min_delta=0.001, metric="rmse", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:  # mode == "max"
            return score > self.best_score + self.min_delta

def prepare_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the California Housing dataset."""
    print("--- 1. Data Preparation ---")
    X_all, y_all = fetch_dataset_from_cgmacros(config['meal_type'])

    rng = np.random.default_rng(config["random_seed"])
    num_samples_to_use = min(config["num_samples_to_use"], len(y_all))
    indices = rng.choice(np.arange(len(y_all)), size=num_samples_to_use, replace=False)
    X, y = X_all[indices], y_all[indices]

    splitter = partial(
        train_test_split,
        test_size=config["valid_set_ratio"],
        random_state=config["random_seed"],
    )
    X_train, X_test, y_train, y_test = splitter(X, y)

    print(
        f"Loaded and split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples."
    )
    print("---------------------------\n")
    return X_train, X_test, y_train, y_test


def setup_regressor(config: dict) -> tuple[TabPFNRegressor, dict]:
    """Initializes the TabPFN regressor and its configuration."""
    print("--- 2. Model Setup ---")
    regressor_config = {
        'categorical_features_indices': [6],
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": config['n_estimators'],
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    regressor = TabPFNRegressor(
        **regressor_config, fit_mode="batched", differentiable_input=False
    )

    print(f"Using device: {config['device']}")
    print("----------------------\n")
    return regressor, regressor_config


def evaluate_regressor(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float, float]:
    """Evaluates the regressor's performance on the test set."""
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_regressor.fit(X_train, y_train)

    try:
        predictions = eval_regressor.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        r = pearsonr(y_test, predictions)[0]
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        rmse, mae, r2, r = np.nan, np.nan, np.nan

    return rmse, mae, r2, r


def main() -> None:
    original_dir = os.getcwd()
    data_dir = original_dir + '/cgmacros1.0/CGMacros'
    
    """Main function to configure and run the finetuning workflow."""
    # --- Master Configuration ---
    # This improved structure separates general settings from finetuning hyperparameters.
    config = {
        "meal_type": 1,
        'n_estimators': 8,
        # Sets the computation device ('cuda' for GPU if available, otherwise 'cpu').
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # The total number of samples to draw from the full dataset. This is useful for
        # managing memory and computation time, especially with large datasets.
        # For very large datasets the entire dataset is preprocessed and then
        # fit in memory, potentially leading to OOM errors.
        "num_samples_to_use": 100_000,
        # A seed for random number generators to ensure that data shuffling, splitting,
        # and model initializations are reproducible.
        "random_seed": 42,
        # The proportion of the dataset to allocate to the valid set for final evaluation.
        "valid_set_ratio": 0.2,
        # During evaluation, this is the number of samples from the training set given to the
        # model as context before it makes predictions on the test set.
        "n_inference_context_samples": 256,
    }
    config["finetuning"] = {
        # The total number of passes through the entire fine-tuning dataset.
        "epochs": 10,
        # A small learning rate is crucial for fine-tuning to avoid catastrophic forgetting.
        "learning_rate": 1.5e-6,
        # Meta Batch size for finetuning, i.e. how many datasets per batch. Must be 1 currently.
        "meta_batch_size": 1,
        # The number of samples within each training data split. It's capped by
        # n_inference_context_samples to align with the evaluation setup.
        "batch_size": int(
            min(
                config["n_inference_context_samples"],
                config["num_samples_to_use"] * (1 - config["valid_set_ratio"]),
            )
        ),
        # Early stopping configuration
        "early_stopping": {
            "patience": 3,  # Number of epochs to wait before stopping
            "min_delta": 0.001,  # Minimum change to qualify as an improvement
            "metric": "rmse",  # Metric to monitor ('rmse', 'mae', 'r2')
            "mode": "min",  # 'min' for rmse/mae, 'max' for r2
        },
        # Gradient clipping configuration
        "gradient_clipping": {
            "max_norm": 1.0,  # Maximum norm for gradients
            "norm_type": 2,  # Type of norm (2 for L2 norm)
        },
    }

    # --- Setup Data, Model, and Dataloader ---
    X_train, X_test, y_train, y_test = prepare_data(config)
    regressor, regressor_config = setup_regressor(config)

    splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
    # Note: `max_data_size` corresponds to the finetuning `batch_size` in the config
    training_datasets = regressor.get_preprocessed_datasets(
        X_train, y_train, splitter, max_data_size=config["finetuning"]["batch_size"]
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )

    # Optimizer must be created AFTER get_preprocessed_datasets, which initializes the model
    optimizer = Adam(
        regressor.model_.parameters(), lr=config["finetuning"]["learning_rate"]
    )
    print(
        f"--- Optimizer Initialized: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
    )

    # Create evaluation config, linking it to the master config
    eval_config = {
        **regressor_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
        },
    }

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")
    
    # Initialize early stopping
    early_stopping_config = config["finetuning"]["early_stopping"]
    early_stopping = EarlyStopping(
        patience=early_stopping_config["patience"],
        min_delta=early_stopping_config["min_delta"],
        metric=early_stopping_config["metric"],
        mode=early_stopping_config["mode"]
    )
    
    # Gradient clipping config
    grad_clip_config = config["finetuning"]["gradient_clipping"]
    
    # Track metrics for early stopping
    metrics_history = []
    
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            # Create a tqdm progress bar to iterate over the dataloader
            progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
            epoch_losses = []
            
            for data_batch in progress_bar:
                optimizer.zero_grad()
                (
                    X_trains_preprocessed,
                    X_tests_preprocessed,
                    y_trains_znorm,
                    y_test_znorm,
                    cat_ixs,
                    confs,
                    raw_space_bardist_,
                    znorm_space_bardist_,
                    _,
                    y_test_raw,
                ) = data_batch

                regressor.raw_space_bardist_ = raw_space_bardist_[0]
                regressor.bardist_ = znorm_space_bardist_[0]
                regressor.fit_from_preprocessed(
                    X_trains_preprocessed, y_trains_znorm, cat_ixs, confs
                )
                logits, _, _ = regressor.forward(X_tests_preprocessed)

                # For regression, the loss function is part of the preprocessed data
                loss_fn = znorm_space_bardist_[0]
                y_target = y_test_znorm

                loss = loss_fn(logits, y_target.to(config["device"])).mean()
                
                # Check for numeric instability
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected at epoch {epoch}: {loss.item()}")
                    continue
                
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    regressor.model_.parameters(),
                    max_norm=grad_clip_config["max_norm"],
                    norm_type=grad_clip_config["norm_type"]
                )
                
                optimizer.step()
                
                epoch_losses.append(loss.item())
                # Set the postfix of the progress bar to show the current loss
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Log average epoch loss
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

        # Evaluation Step (runs before finetuning and after each epoch)
        rmse, mae, r2, r = evaluate_regressor(
            regressor, eval_config, X_train, y_train, X_test, y_test
        )

        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(
            f"📊 {status} Evaluation | Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}, Test R: {r:.4f}"
        )
        
        # Store metrics for history
        metrics_dict = {"epoch": epoch, "rmse": rmse, "mae": mae, "r2": r2, "r": r}
        metrics_history.append(metrics_dict)
        
        # Early stopping check (skip for initial evaluation)
        if epoch > 0:
            # Get the metric to monitor
            monitor_metric = metrics_dict[early_stopping_config["metric"]]
            
            # Check early stopping
            if early_stopping(monitor_metric, epoch):
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                print(f"Best {early_stopping_config['metric']}: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
                break

    print("--- ✅ Finetuning Finished ---")
    # --- 4. Save logs and final model ---
    print("--- 4. Saving Model and Logs ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(original_dir, "logs")
    models_dir = os.path.join(original_dir, "models")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    config['timestamp'] = timestamp
    config['regressor_config'] = regressor_config
    config['final_metrics'] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "r": float(r)}
    config['early_stopping'] = {
        "triggered": early_stopping.early_stop,
        "best_epoch": early_stopping.best_epoch,
        "best_score": early_stopping.best_score,
        "final_epoch": len(metrics_history) - 1,
    }
    config['metrics_history'] = metrics_history
    
    metadata = config

    log_path = os.path.join(logs_dir, f"finetune_log_{timestamp}.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved finetune log to: {log_path}")
    except Exception as e:
        print(f"Failed to save finetune log: {e}")

    # Save model: state_dict and optimizer state for reproducibility
    model_state_path = os.path.join(models_dir, f"tabpfn_model_state_{timestamp}.pt")
    try:
        torch.save(
            {
                "model_state_dict": regressor.model_.state_dict(),
                "regressor_config": regressor_config,
                "optimizer_state_dict": optimizer.state_dict(),
                "metadata": metadata,
                "datasets": (X_train, X_test, y_train, y_test)
            },
            model_state_path,
        )
        print(f"Saved model state to: {model_state_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")



if __name__ == "__main__":
    main()
