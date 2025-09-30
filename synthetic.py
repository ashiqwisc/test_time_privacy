# Baseline with synthetic data
import numpy as np
import torch
import argparse
from torchvision import transforms
import random
import yaml
import copy
import os
import pytz
import time
from load_dataset import load_data
from uniformity_helper import data_splitter, change_model_params
from train import ModelTrainer
from models import ModelInitializer
from evaluator import evaluate, forget_quality
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

def sample_ball_around(x, eps, num_samples):
    """
    Sample points within an L2 ball of radius eps around x.
    Handles both (H, W) and (C, H, W) input shapes.
    """
    x = x.to(torch.float32)
    original_shape = x.shape
    x_flat = x.flatten()  # (D,)
    dim = x_flat.shape[0]

    # Sample normalized noise vectors
    noise = torch.randn((num_samples, dim), device=x.device)
    noise = noise / noise.norm(dim=1, keepdim=True).clamp(min=1e-12)

    # Sample radii uniformly from the volume of an L2 ball
    radii = torch.rand(num_samples, device=x.device).pow(1.0 / dim) * eps
    noise = noise * radii.unsqueeze(1)

    # Add noise to original input and reshape
    x_noisy = x_flat.unsqueeze(0) + noise
    x_noisy = x_noisy.view(num_samples, *original_shape)

    # Clamp to valid image range
    return x_noisy.clamp(0, 1)

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
        # guarantee y is a Tensor
        return x, torch.tensor(y, dtype=torch.long)

# Augmented forget dataset
class AugmentedForgetDataset(Dataset):
    def __init__(self, original_dataset, synthetic_data, synthetic_labels, transform=None):
        self.original_dataset = original_dataset
        self.synthetic_data = synthetic_data
        self.synthetic_labels = synthetic_labels
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset) + len(self.synthetic_data)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            x, y = self.original_dataset[idx]
            # if the original dataset has its own transform, apply it
            if self.transform:
                x = self.transform(x)
            # convert the python int label into a tensor
            y = torch.tensor(y, dtype=torch.long)
        else:
            x = self.synthetic_data[idx - len(self.original_dataset)]
            y = self.synthetic_labels[idx - len(self.original_dataset)]
            if self.transform:
                x = self.transform(x)
            y = torch.tensor(y, dtype=torch.long)
        return x, y

# Augmented the forget set DataLoader (be sure to preseve transforms)
def augment_forget_loader(forget_train_loader, num_classes, eps, num_samples_per_input, device):
    augmented_x = []
    augmented_y = []

    for x_batch, _ in forget_train_loader:
        x_batch = x_batch.to(device)
        for x in x_batch:
            synthetic = sample_ball_around(x, eps, num_samples_per_input)
            labels = torch.randint(low=0, high=num_classes, size=(num_samples_per_input,), device=device)
            augmented_x.append(synthetic)
            augmented_y.append(labels)

    if augmented_x:
        augmented_x = torch.cat(augmented_x, dim=0).cpu()
        augmented_y = torch.cat(augmented_y, dim=0).cpu()

        original_dataset = forget_train_loader.dataset
        transform = getattr(original_dataset, 'transform', None)

        augmented_dataset = AugmentedForgetDataset(
            original_dataset,
            synthetic_data=augmented_x,
            synthetic_labels=augmented_y,
            transform=transform
        )

        forget_train_loader = DataLoader(augmented_dataset, batch_size=forget_train_loader.batch_size, shuffle=True)

    return forget_train_loader


# Return only the synthetic data loader, not including the original train loader
def synthetic_loader(forget_train_loader, num_classes, eps, num_samples_per_input, device):
    """
    Generate a new forget loader containing only synthetic samples drawn from L2 balls around
    the original forget instances (original instances are dropped).
    """
    augmented_x = []
    augmented_y = []

    # Generate synthetic samples around each forget instance
    for x_batch, _ in forget_train_loader:
        x_batch = x_batch.to(device)
        for x in x_batch:
            synthetic = sample_ball_around(x, eps, num_samples_per_input)
            labels = torch.randint(low=0, high=num_classes, size=(num_samples_per_input,), device=device)
            augmented_x.append(synthetic)
            augmented_y.append(labels)

    if not augmented_x:
        # If no synthetic samples, return an empty loader
        empty_ds = TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
        return DataLoader(empty_ds, batch_size=forget_train_loader.batch_size, shuffle=True)

    # Concatenate all synthetic samples
    augmented_x = torch.cat(augmented_x, dim=0).cpu()
    augmented_y = torch.cat(augmented_y, dim=0).cpu()

    # Wrap synthetic tensors into a dataset with transforms if any
    transform = getattr(forget_train_loader.dataset, 'transform', None)
    base_synth_ds = TensorDataset(augmented_x, augmented_y)
    synth_ds = TensorLabelDataset(base_synth_ds, transform=transform)

    # Return loader over synthetic-only dataset
    return DataLoader(synth_ds, batch_size=forget_train_loader.batch_size, shuffle=True)


def save_results_to_file(results_data, results_file_path):
    """
    Save experiment results to a text file with fields in a specific order.
    
    Args:
        results_data: Dictionary containing the results
        results_file_path: Path to save the results file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    
    # Define the order of fields - grouped by model type
    fields = [
        "Dataset",
        "Model",
        "Lambda",
        "Size of Ball",
        "Number of Ball Samples",
        "Learning Rate",
        "Synthetic model retain acc",
        "Synthetic model test acc",
        "Synthetic model forget acc, original train dataset",
        "Synthetic model forget acc, augmented train dataset",
        "Synthetic model avg. uniformity loss over forget set, original train dataset",
        "Synthetic model avg. uniformity loss over forget set, augmented train dataset",
        "Synthetic model avg. max{0, f(x)_t-1/|Y|}, original train dataset" ,
        "Synthetic model avg. max{0, f(x)_t-1/|Y|}, augmented train dataset" ,
        "Synthetic model duration"
        ""
    ]
    
    with open(results_file_path, 'w') as f:
        # Write each field on a separate line
        for field in fields:
            value = results_data.get(field, "-")
            # Format floats with 2 decimal places
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            f.write(f"{field}: {formatted_value}\n")
        
    print(f"Results saved to: {results_file_path}")
    
# Add this function to collect results from the evaluations
def collect_results(dataset, model_name, 
                   synthetic_model,
                   retain_loader, test_loader, forget_loader_original, forget_loader_augmented, 
                   synthetic_dur,
                   args):
    """
    Collect all results for the experiment in a dictionary.
    
    Returns:
        Dictionary with all the result metrics
    """
    # Get all required metrics
    synthetic_retain_acc = evaluate(synthetic_model, retain_loader, device)
    synthetic_test_acc = evaluate(synthetic_model, test_loader, device)
    synthetic_forget_acc_original = evaluate(synthetic_model, forget_loader_original, device)
    synthetic_forget_acc_augmented = evaluate(synthetic_model, forget_loader_augmented, device)
    
    # Get forget metrics for pretrained model
    synthetic_forget_metrics_original = forget_quality(
        synthetic_model, forget_loader_original, 
        args.forget_images_path, "synthetic_model", 
        args.batch_size, args.num_classes, args, device
    )


    synthetic_forget_metrics_augmented = forget_quality(
            synthetic_model, forget_loader_augmented, 
            args.forget_images_path, "synthetic_model", 
            args.batch_size, args.num_classes, args, device
        )
    
    if args.with_reg: 
        convex_approx_lambda = args.convex_approx_lambda
    else: 
        convex_approx_lambda = 0
    
    # Compile all results in a dictionary
    results = {
        "Dataset": dataset,
        "Model": model_name,
        "Lambda": convex_approx_lambda,
        "Size of Ball": args.ball_size,
        "Number of Ball Samples": args.ball_samples,
        "Learning Rate": args.lr,
        "Synthetic model retain acc": synthetic_retain_acc,
        "Synthetic model test acc": synthetic_test_acc,
        "Synthetic model forget acc, original train dataset": synthetic_forget_acc_original,
        "Synthetic model forget acc, augmented train dataset": synthetic_forget_acc_augmented,
        "Synthetic model avg. uniformity loss over forget set, original train dataset": synthetic_forget_metrics_original['avg_unif_loss'],
        "Synthetic model avg. uniformity loss over forget set, augmented train dataset": synthetic_forget_metrics_augmented['avg_unif_loss'],
        "Synthetic model avg. max{0, f(x)_t-1/|Y|}, original train dataset":  synthetic_forget_metrics_original['avg_confidence_distance'],
        "Synthetic model avg. max{0, f(x)_t-1/|Y|}, augmented train dataset":  synthetic_forget_metrics_augmented['avg_confidence_distance'],
        "Synthetic model duration": synthetic_dur
    }
    
    return results

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def config_to_args(config):
    """Convert a configuration dictionary to a Namespace object like argparse"""
    args = argparse.Namespace()
    
    # training parameters
    args.batch_size = config['training']['batch_size']
    args.epochs = config['training']['epochs']
    args.lr = config['training']['lr']
    args.dataset = config['training']['dataset']
    args.model_type = config['training']['model_type']
    args.train = config['training']['train']
    args.num_layer = config['training']['num_layer']
    args.hidden_dim = config['training']['hidden_dim']
    args.num_classes = config['training']['num_classes']
    args.seed = config['training']['seed']
    
    # model paths
    args.pretrained_model_load_path = config['paths']['trained_model_load_path']
    args.forget_images_path = config['paths']['forget_images_path']
    args.results_file_path = config['paths'].get('results_file_path', './results/experiment_results.txt')  
    
    # uniformity parameters
    args.num_unif = config['uniformity']['num_unif']
    args.unif_batch_size = config['uniformity']['unif_batch_size']
    
    # misc opt parameters
    args.C = config['opt']['C']
    args.with_pgd = config['opt']['with_pgd']
    args.convex_approx_lambda = config['opt']['convex_approx_lambda']
    
    # Synthetic parameters
    args.ball_size = config['synthetic']['ball_size']
    args.ball_samples = config['synthetic']['ball_samples']

    # misc options 
    args.unif_loss = config['options']['unif_loss'] # kl_forward, kl_reverse, square
    args.with_reg = config['options']['with_reg'] 
    
    return args

if __name__ == "__main__":
    print("PID:", os.getpid())
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()
    
     # load configuration from file
    config_path = args.config
    if os.path.exists(config_path):
        config = load_config(config_path)

        # override seed if passed via command line
        if args.seed is not None:
            config['training']['seed'] = args.seed

        # now convert config to args
        args = config_to_args(config)
    else:
        print(f"Error: Config file {config_path} not found.")
        exit(1)
        
        
    use_cuda = torch.cuda.is_available() # use cuda whenever it is available 
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device:{device}")
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    """Load dataset and select model."""
    if args.dataset == "MNIST" or args.dataset == "FMNIST" or args.dataset == "KMNIST": 
        data_shape = [3,28,28]
    if args.dataset == "CIFAR100" or args.dataset == "CIFAR10" or args.dataset == "CIFAR5" or args.dataset == "SVHN_standard" or args.dataset == "SVHN_full":
        data_shape = [3,32,32]
    if args.dataset == "STL10" or args.dataset == "STL5":
        data_shape = [3,96,96]


    # Make data loader deterministic
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    num_workers = 0 

    """Load data"""
    train_loader, test_loader, train_eval_loader, train_dataset, test_dataset = load_data(args, device, num_workers=num_workers, generator=g, worker_init_fn=seed_worker)

    """Obtain forget set and retain set"""
    retain_train_loader, retain_eval_loader, forget_train_loader, forget_eval_loader = data_splitter(train_loader, train_eval_loader, args)

    """Sample ball instances, assign random labels, and create a new loader using this--replace the forget loader, don't append!"""

    synthetic_loader_new = synthetic_loader(forget_train_loader,
                                      num_classes=args.num_classes, 
                                      eps=args.ball_size, 
                                      num_samples_per_input=args.ball_samples, 
                                      device=device)

    # Ensure everything tensors
    retain_wrapped = TensorLabelDataset(
        retain_train_loader.dataset,
        transform=getattr(retain_train_loader.dataset, 'transform', None)
    )
    synthetic_wrapped = TensorLabelDataset(
        synthetic_loader_new.dataset,
        transform = getattr(synthetic_loader_new.dataset, 'transform', None)

    )
    combined_dataset = ConcatDataset([retain_wrapped, synthetic_wrapped])
    combined_train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
    
    # combine evaluation dataset with old forget set
    forget_eval_loader_new =  augment_forget_loader(
        forget_eval_loader,
        num_classes=args.num_classes,
        eps=args.ball_size,
        num_samples_per_input=args.ball_samples,
        device=device
    )

    retain_eval_wrapped = TensorLabelDataset(
        retain_eval_loader.dataset,
        transform=getattr(retain_train_loader.dataset, 'transform', None)
    )
    forget_eval_wrapped = TensorLabelDataset(
        forget_eval_loader_new.dataset,
        transform=getattr(forget_eval_loader_new.dataset, 'transform', None)
    )

    combined_eval_dataset = ConcatDataset([retain_eval_wrapped, forget_eval_wrapped])
    combined_eval_loader = DataLoader(combined_eval_dataset, batch_size=args.batch_size, shuffle=True)

    """Train synthetic model over joint synthetic and retain loaders"""

    initalizer = ModelInitializer(device)
    model = initalizer.init_model(args)
    trainer = ModelTrainer(device)

    trainer = ModelTrainer(device)
    synthetic_start = time.time()
    synthetic_model =  trainer.retrain_from_scratch(model=model, 
                            retain_loader=combined_train_loader,   
                            args=args, 
                            num_workers=num_workers, 
                            seed_worker=seed_worker, 
                            g=g,
                            data_shape=data_shape,
                            logistic=False)
    synthetic_dur = time.time() - synthetic_start 

    """Evaluate synthetic model"""
    print("-------------------------------------Evaluating synthetic model on original training set------------------------------------------------------------")
    print(f"Train accuracy: {evaluate(synthetic_model, train_eval_loader, device)}")

    print("-------------------------------------Evaluating synthetic model on augmented training set------------------------------------------------------------")
    print(f"Train accuracy: {evaluate(synthetic_model, combined_eval_loader, device)}")

    print("-------------------------------------Evaluating pretrained model on retain set and test set------------------------------------------------------------")
    print(f"Retain set accuracy: {evaluate(synthetic_model, retain_eval_loader, device)}")
    print(f"Test set accuracy: {evaluate(synthetic_model, test_loader, device)}")

    print("-------------------------------------Evaluating synthetic model on original forget set-----------------------------------------------------------")
    forget_quality(synthetic_model, forget_eval_loader, 
               args.forget_images_path, "synthetic_model", args.batch_size, args.num_classes, args)

    print("-------------------------------------Evaluating synthetic model on augmented forget set-----------------------------------------------------------")
    forget_quality(synthetic_model, forget_eval_loader_new, 
               args.forget_images_path, "synthetic_model", args.batch_size, args.num_classes, args)
    
    """Save results"""
    # Collect and save results
    results = collect_results(
        dataset=args.dataset, 
        model_name=args.model_type, 
        synthetic_model=synthetic_model,
        retain_loader=retain_eval_loader, 
        test_loader=test_loader, 
        forget_loader_original=forget_eval_loader, 
        forget_loader_augmented=forget_eval_loader_new, 
        synthetic_dur=synthetic_dur,
        args=args
    )

    # Generate timestamp for unique filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%m_%d_%H_%M")

    # Create a unique filename for the results
    results_filename = f"{args.model_type}_{args.ball_size}_{timestamp}.txt"
    results_path = os.path.join(os.path.dirname(args.results_file_path), results_filename)

    # Save results to file
    save_results_to_file(results, results_path)