import numpy as np
import torch
import argparse
import yaml
import os
import time
import random
from load_dataset import load_data
from uniformity_helper import data_splitter
from train import ModelTrainer
from models import ModelInitializer
from evaluator import evaluate, forget_quality
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from datetime import datetime
import copy

def add_gaussian_noise(x, std_dev):
    noise = torch.randn_like(x) * std_dev
    noisy_x = x + noise
    return torch.clamp(noisy_x, 0, 1)

class TensorLabelDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform:
            x = self.transform(x)
        # Guarantee y is a Tensor
        return x, torch.tensor(y, dtype=torch.long)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def config_to_args(config):
    args = argparse.Namespace()
    for section in config:
        for key, value in config[section].items():
            setattr(args, key, value)
    return args
# --- End of unchanged helper functions ---


def create_modified_sets(original_loader, std_dev, num_classes, device):
    """
    Creates two modified versions of the original dataset:
    1. D' (D_prime): Original images perturbed with Gaussian noise, original labels.
    2. D~ (D_tilde): Original images, labels sampled uniformly at random.
    
    Returns:
        Four Tensors: (noisy_x, original_y, original_x, random_y)
    """
    original_x_list = []
    original_y_list = []
    noisy_x_list = []
    random_y_list = []

    for x_batch, y_batch in original_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # For D'
        noisy_x = add_gaussian_noise(x_batch, std_dev)
        noisy_x_list.append(noisy_x.cpu())
        
        # For D~
        random_y = torch.randint(low=0, high=num_classes, size=(x_batch.size(0),), device=device)
        random_y_list.append(random_y.cpu())
        
        # Keep original data for both sets
        original_x_list.append(x_batch.cpu())
        original_y_list.append(y_batch.cpu())

    # Concatenate all batches
    noisy_x_tensor = torch.cat(noisy_x_list, dim=0)
    original_y_tensor = torch.cat(original_y_list, dim=0)
    original_x_tensor = torch.cat(original_x_list, dim=0)
    random_y_tensor = torch.cat(random_y_list, dim=0)
    
    return noisy_x_tensor, original_y_tensor, original_x_tensor, random_y_tensor


def save_results_to_file(results_data, results_file_path):
    """Saves the comprehensive evaluation results to a text file."""
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    
    # Define a more comprehensive set of fields
    fields = [
        "Dataset", "Model", "Lambda", "Gaussian Noise Std Dev", "Learning Rate", "Duration (s)",
        # --- Accuracy Metrics ---
        "D_f Accuracy", "D_r Accuracy",
        "D_f' Accuracy", "D_r' Accuracy",
        "D_f~ Accuracy", "D_r~ Accuracy",
        "Test Set Accuracy",
        "Test Set' Accuracy", # Added field for perturbed test set
        # --- Confidence Distance Metrics ---
        "D_f Conf-Dist", "D_r Conf-Dist",
        "D_f' Conf-Dist", "D_r' Conf-Dist",
        "D_f~ Conf-Dist", "D_r~ Conf-Dist"
    ]
    
    with open(results_file_path, 'w') as f:
        f.write("--- Experiment Details ---\n")
        for field in fields[:6]:
            value = results_data.get(field, "-")
            f.write(f"{field}: {value}\n")
        
        f.write("\n--- Evaluation Metrics ---\n")
        for field in fields[6:]:
            value = results_data.get(field, "-")
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            f.write(f"{field}: {formatted_value}\n")
        
    print(f"Results saved to: {results_file_path}")


def collect_all_results(model, datasets, dataset_names, test_loader, test_prime_loader, duration, args, device):
    """Collects evaluation metrics across all specified datasets."""
    results = {
        "Dataset": args.dataset,
        "Model": args.model_type,
        "Lambda": args.convex_approx_lambda if args.with_reg else 0,
        "Gaussian Noise Std Dev": args.std_dev,
        "Learning Rate": args.lr,
        "Duration (s)": duration
    }
    
    # Evaluate each dataset
    for name, loader in zip(dataset_names, datasets):
        acc = evaluate(model, loader, device)
        # Using forget_quality for confidence distance calculation on all sets
        metrics = forget_quality(
            model, loader, args.forget_images_path, f"temp_{name}",
            args.batch_size, args.num_classes, args, device
        )
        
        results[f"{name} Accuracy"] = acc
        results[f"{name} Conf-Dist"] = metrics['avg_confidence_distance']

    # Also get test set and perturbed test set accuracy
    results["Test Set Accuracy"] = evaluate(model, test_loader, device)
    results["Test Set' Accuracy"] = evaluate(model, test_prime_loader, device) # Added evaluation for D_test'
    
    return results


if __name__ == "__main__":
    print("PID:", os.getpid())
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_gaussian.yaml', help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config')
    
    cli_args = parser.parse_args()
    
    if not os.path.exists(cli_args.config):
        print(f"Error: Config file {cli_args.config} not found.")
        exit(1)
        
    config = load_config(cli_args.config)
    if cli_args.seed is not None:
        config['training']['seed'] = cli_args.seed
    args = config_to_args(config)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Make data loading deterministic
    g = torch.Generator()
    g.manual_seed(args.seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    num_workers = 2 if use_cuda else 0

    """1. Load original data D"""
    train_loader, test_loader, train_eval_loader, _, _ = load_data(args, device, num_workers=num_workers, generator=g, worker_init_fn=seed_worker)

    """2. Split D into D_f and D_r"""
    retain_train_loader, retain_eval_loader, forget_train_loader, forget_eval_loader = data_splitter(train_loader, train_eval_loader, args)

    """3. Create modified datasets D', D~ from D_f and D_r, and D_test'"""
    print("Creating modified datasets D' (noisy) and D~ (random labels)...")
    
    # Create D_r' and D_r~ from D_r
    noisy_x_r, original_y_r, original_x_r, random_y_r = create_modified_sets(
        retain_train_loader, args.std_dev, args.num_classes, device
    )
    # Create D_f' and D_f~ from D_f
    noisy_x_f, original_y_f, original_x_f, random_y_f = create_modified_sets(
        forget_train_loader, args.std_dev, args.num_classes, device
    )
    
    # Create perturbed test set D_test'
    noisy_x_test_list, original_y_test_list = [], []
    for x_batch, y_batch in test_loader:
        noisy_x = add_gaussian_noise(x_batch.to(device), args.std_dev)
        noisy_x_test_list.append(noisy_x.cpu())
        original_y_test_list.append(y_batch.cpu())
    test_prime_dataset = TensorDataset(torch.cat(noisy_x_test_list, dim=0), torch.cat(original_y_test_list, dim=0))
    test_prime_eval_loader = DataLoader(test_prime_dataset, batch_size=args.batch_size, shuffle=False)
    print("Created perturbed test set D_test'.")


    # --- Prepare datasets for training ---
    # D' = D_f' U D_r'
    dr_prime_dataset = TensorDataset(noisy_x_r, original_y_r)
    df_prime_dataset = TensorDataset(noisy_x_f, original_y_f)
    
    # D~ = D_f~ U D_r~
    dr_tilde_dataset = TensorDataset(original_x_r, random_y_r)
    df_tilde_dataset = TensorDataset(original_x_f, random_y_f)

    # Training set is D' U D~
    training_dataset = ConcatDataset([dr_prime_dataset, df_prime_dataset, dr_tilde_dataset, df_tilde_dataset])
    combined_train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    """4. Train model f on D' U D~ or load from path """
    initializer = ModelInitializer(device)
    model = initializer.init_model(args)
    
    # Load a pre-trained model if a path is specified
    if hasattr(args, 'trained_model_load_path') and args.trained_model_load_path != "" and os.path.exists(args.trained_model_load_path):
        print(f"--- Loading pre-trained model from: {args.trained_model_load_path} ---")
        model.load_state_dict(torch.load(args.trained_model_load_path, map_location=device))
        trained_model = model
        duration = 0.0
    else:
        # Otherwise, train a new model from scratch
        print(f"--- Training model from scratch on D' U D~ (size: {len(training_dataset)}) ---")
        trainer = ModelTrainer(device)
        start_time = time.time()
        trained_model = trainer.retrain_from_scratch(
            model=model, 
            retain_loader=combined_train_loader,   
            retain_val_loader=retain_eval_loader, # Using retain set for validation
            args=args, 
            num_workers=num_workers, 
            seed_worker=seed_worker, 
            g=g,
            logistic=False
        )
        duration = time.time() - start_time 
        print(f"Training complete. Duration: {duration:.2f}s")

        # Save the newly trained model if a save path is provided
        if hasattr(args, 'trained_model_save_path') and args.trained_model_save_path != "":
            save_dir = os.path.dirname(args.trained_model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Use the full path from config for saving
            path = args.trained_model_save_path 
            torch.save(trained_model.state_dict(), path)
            print(f"--- Model saved to: {path} ---")


    """5. Evaluate f on all 6 partitions + test set + perturbed test set"""
    print("\n--- Evaluating Model on All Data Partitions ---")

    # For evaluation, we need DataLoaders for all sets
    dr_prime_eval_loader = DataLoader(dr_prime_dataset, batch_size=args.batch_size, shuffle=False)
    df_prime_eval_loader = DataLoader(df_prime_dataset, batch_size=args.batch_size, shuffle=False)
    dr_tilde_eval_loader = DataLoader(dr_tilde_dataset, batch_size=args.batch_size, shuffle=False)
    df_tilde_eval_loader = DataLoader(df_tilde_dataset, batch_size=args.batch_size, shuffle=False)
    
    datasets_to_evaluate = [
        forget_eval_loader, retain_eval_loader,
        df_prime_eval_loader, dr_prime_eval_loader,
        df_tilde_eval_loader, dr_tilde_eval_loader
    ]
    dataset_names = [
        "D_f", "D_r", 
        "D_f'", "D_r'", 
        "D_f~", "D_r~"
    ]

    # Print a quick summary
    for name, loader in zip(dataset_names, datasets_to_evaluate):
        acc = evaluate(trained_model, loader, device)
        print(f"Accuracy on {name}: {acc:.4f}")
    print(f"Accuracy on Test Set: {evaluate(trained_model, test_loader, device):.4f}")
    print(f"Accuracy on Test Set' (Perturbed): {evaluate(trained_model, test_prime_eval_loader, device):.4f}")


    """6. Save Comprehensive Results"""
    results = collect_all_results(
        model=trained_model,
        datasets=datasets_to_evaluate,
        dataset_names=dataset_names,
        test_loader=test_loader,
        test_prime_loader=test_prime_eval_loader, # Pass the new loader
        duration=duration,
        args=args,
        device=device
    )

    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    results_filename = f"{args.model_type}_{args.dataset}_{args.std_dev}_{timestamp}.txt"
    results_dir = os.path.dirname(args.results_file_path) if '.' in os.path.basename(args.results_file_path) else args.results_file_path
    results_path = os.path.join(results_dir, results_filename)
    
    save_results_to_file(results, results_path)