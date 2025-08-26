from functools import partial
import numpy as np
import torch
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
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 2,
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
        "meal_type": 5,
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
        "epochs": 8,
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
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            # Create a tqdm progress bar to iterate over the dataloader
            progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
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
                loss.backward()
                optimizer.step()

                # Set the postfix of the progress bar to show the current loss
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Evaluation Step (runs before finetuning and after each epoch)
        rmse, mae, r2, r = evaluate_regressor(
            regressor, eval_config, X_train, y_train, X_test, y_test
        )

        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(
            f"📊 {status} Evaluation | Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}\n, Test R: {r:.4f}"
        )

    print("--- ✅ Finetuning Finished ---")


if __name__ == "__main__":
    main()
