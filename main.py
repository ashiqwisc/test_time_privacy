import numpy as np
import torch
import argparse
import random
import yaml
import copy
import os
import time
from datetime import datetime
from load_dataset import load_data
from uniformity_exact import exact_certified_uniformity
from uniformity_helper import data_splitter
from train import ModelTrainer
from models import ModelInitializer
from evaluator import evaluate, forget_quality
from attack import create_perturbed_dataloader
from nearestneighbors import get_nearest_neighbors_dataloader
from testsampling import test_sampling
import pytz

import sys
import torch.nn as nn
from vit_trainer_tinyimagenet import ViTTrainer
from vit_trainer_cifar import CIFAR100ViTTrainer

class TeeStdout:
    def __init__(self, terminal, log_file, print_to_terminal=False):
        self.terminal = terminal
        self.log = log_file
        self.print_to_terminal = print_to_terminal
    
    def write(self, message):
        if "batch/s" not in message and "|" not in message:
            if self.print_to_terminal:
                self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        if self.print_to_terminal:
            self.terminal.flush()
        self.log.flush()

def setup_environment(args):
    """Setup device, seeds, and deterministic behavior"""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
    return device

def get_data_shape(args):
    """Get data shape based on dataset and model type"""
    # # Handle MedMNIST datasets
    # if args.dataset.lower() in [name.lower() for name in INFO.keys()]:
    #     medmnist_name = None
    #     for name in INFO.keys():
    #         if name.lower() == args.dataset.lower():
    #             medmnist_name = name
    #             break
    #     if medmnist_name:
    #         info = INFO[medmnist_name]
    #         channels = info['n_channels']
    #         return [channels, 28, 28]
    
    # Handle standard datasets
    if args.dataset in ["MNIST", "FMNIST", "KMNIST"]: 
        return [3, 28, 28]
    elif args.dataset in ["CIFAR100", "CIFAR10", "CIFAR5", "SVHN_standard", "SVHN_full"]:
        if args.model_type in ["ViT-S_16", "ViT-B_16"] or 'vit' in args.model_type.lower():
            return [3, 224, 224]  # ViT resizes to 224
        else:
            return [3, 32, 32]
    elif args.dataset in ["STL10", "STL5"]:
        return [3, 96, 96]
    elif args.dataset == "TinyImageNet":
        if 'vit' in args.model_type.lower():
            return [3, 224, 224]  # ViT resizes to 224
        else:
            return [3, 64, 64]
    else:
        return [3, 32, 32]  # Default

def load_or_train_pretrained_model(args, device, train_loader, train_eval_loader, test_loader, data_shape, num_workers, seed_worker, g):
    initializer = ModelInitializer(device)
    
    
    if args.trained_model_load_path:
        print("Loading pretrained model from:", args.trained_model_load_path)
        model = initializer.init_model(args)
        model.load_state_dict(torch.load(args.trained_model_load_path))
        duration = 0
    elif args.continue_train_load_path:
        print("Loading model to continue training from:", args.continue_train_load_path)
        start_time = time.time()
        model = initializer.init_model(args)
        model.load_state_dict(torch.load(args.continue_train_load_path))
        trainer = ModelTrainer(device)
        model = trainer.retrain_from_scratch(
            model=model, retain_loader=train_loader, retain_val_loader=train_eval_loader,
            args=args, num_workers=num_workers, seed_worker=seed_worker, g=g,
            data_shape=data_shape, logistic=False, test_loader=test_loader
        )
        duration = time.time() - start_time
    else:
        print("Training new model from scratch")
        start_time = time.time()
        
        # Choose appropriate trainer based on dataset and model
        if args.dataset == "TinyImageNet" and 'vit' in args.model_type.lower():
            print("using TinyImageNet ViT trainer")
            trainer = ViTTrainer(device)
            model = trainer.retrain_vit_from_scratch(
                retain_loader=train_loader,
                retain_val_loader=train_eval_loader,
                args=args, test_loader=test_loader
            )
        elif args.dataset == "CIFAR100" and args.model_type in ["ViT-S_16", "ViT-B_16"]:
            print("using CIFAR-100 ViT trainer")
            trainer = CIFAR100ViTTrainer(device)
            model = trainer.retrain_cifar100_vit_from_scratch(
                retain_loader=train_loader,
                retain_val_loader=train_eval_loader,
                args=args
            )
        else:
            print("using standard trainer")
            model = initializer.init_model(args)
            print(data_shape)
            trainer = ModelTrainer(device)
            
            def flat_checksum(params):
                with torch.no_grad():
                    return float(torch.cat([p.detach().float().flatten() for p in params if p.requires_grad]).sum().cpu())

            print("=== REPRO CHECK ===")
            xb, yb = next(iter(train_loader))
            print("Batch shape:", tuple(xb.shape), "sum:", float(xb.sum().cpu()), "labels sum:", int(yb.sum().cpu()))
            print("Param count:", sum(p.numel() for p in model.parameters()))
            print("Params checksum:", flat_checksum(model.parameters()))
            model.train()
            out = model(xb.to(next(model.parameters()).device))
            loss = torch.nn.CrossEntropyLoss()(out, yb.to(next(model.parameters()).device))
            print("Initial loss:", float(loss.detach().cpu()))
            print("===================")
            
            
            
            
            model = trainer.retrain_from_scratch(
                model=model, retain_loader=train_loader, retain_val_loader=train_eval_loader,
                args=args, num_workers=num_workers, seed_worker=seed_worker, g=g,
                data_shape=data_shape, logistic=False
            )
        
        duration = time.time() - start_time
    
    # Save model if path specified
    if args.trained_model_save_path:
        os.makedirs(args.trained_model_save_path, exist_ok=True)
        save_path = os.path.join(args.trained_model_save_path, f"pretrained_model_{args.dataset}_{args.model_type}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved pretrained model to: {save_path}")
    
    return model, duration

def run_warmup_training(pretrained_model, forget_train_loader, retain_train_loader, retain_eval_loader, forget_eval_loader, test_loader, args, device, seed_worker, g):
    """Run warmup training if enabled"""
    if not args.warmup:
        return pretrained_model, 0
    
    print("Running warmup Pareto training...")
    start_time = time.time()
    
    trainer = ModelTrainer(device)
    warmup_model, _ = trainer.train_pareto_model(
        model=copy.deepcopy(pretrained_model),
        forget_loader=forget_train_loader,
        retain_loader=retain_train_loader,
        retain_val_loader=retain_eval_loader,
        forget_val_loader=forget_eval_loader,
        test_loader=test_loader,
        device=device,
        args=args,
        seed_worker=seed_worker,
        g=g,
        reg=args.warmup_reg
    )
    
    duration = time.time() - start_time
    print(f"Warmup duration: {duration:.2f} seconds")
    
    return warmup_model, duration

def run_uniformity_certification(model, retain_eval_loader, forget_eval_loader, args, device):
    """Run uniformity certification based on specified mode"""
    if not args.run_uniformity:
        return model, 0
    
    print(f"Running uniformity certification (mode: {args.uniformity_mode})...")
    start_time = time.time()
    
    if args.uniformity_mode == "exact":
        certified_model = exact_certified_uniformity(
            model, retain_eval_loader, forget_eval_loader, args, device, args.util_unif_tradeoff_theta
        )
    else:
        print(f"Uniformity mode '{args.uniformity_mode}' not implemented, skipping")
        certified_model = model
    
    duration = time.time() - start_time
    print(f"Uniformity certification duration: {duration:.2f} seconds")
    
    return certified_model, duration

def run_retrain_baseline(args, device, retain_train_loader, retain_eval_loader, test_loader, data_shape, num_workers, seed_worker, g):
    """Run retrain-from-scratch baseline"""
    if not args.run_retrain:
        return None, 0
    
    print("Training retrain baseline (without forget set)...")
    start_time = time.time()
    if args.dataset == "TinyImageNet" and 'vit' in args.model_type.lower():
        trainer = ViTTrainer(device)
        model = trainer.retrain_vit_from_scratch(
            retain_loader=retain_train_loader, retain_val_loader=retain_eval_loader,
            args=args, test_loader=test_loader
        )
    elif args.dataset == "CIFAR100" and args.model_type in ["ViT-S_16", "ViT-B_16"]:
        trainer = CIFAR100ViTTrainer(device)
        model = trainer.retrain_cifar100_vit_from_scratch(
            retain_loader=retain_train_loader, retain_val_loader=retain_eval_loader,
            args=args, test_loader=test_loader
        )
    else:
        initializer = ModelInitializer(device)
        model = initializer.init_model(args)
        trainer = ModelTrainer(device)
        model = trainer.retrain_from_scratch(
            model=model, retain_loader=retain_train_loader, retain_val_loader=retain_eval_loader,
            args=args, num_workers=num_workers, seed_worker=seed_worker, g=g,
            data_shape=data_shape, logistic=False
        )
    
    duration = time.time() - start_time
    print(f"Retrain duration: {duration:.2f} seconds")
    
    # Save model if path specified
    if args.retrain_save_path:
        os.makedirs(args.retrain_save_path, exist_ok=True)
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        save_path = os.path.join(args.retrain_save_path, f"retrain_model_{args.dataset}_{args.model_type}.pth")
        torch.save(model_state, save_path)
        print(f"Saved retrain model to: {save_path}")
    
    return model, duration

def run_pareto_finetuning(base_model, forget_train_loader, retain_train_loader, retain_eval_loader, forget_eval_loader, test_loader, args, device, seed_worker, g):
    print( args.run_pareto)
    if not args.run_pareto:
        return None, 0, None

    if args.unif_model_load_path:
        print("Loading Pareto model from:", args.unif_model_load_path)
        initializer = ModelInitializer(device)
        pareto_model = initializer.init_model(args)
        pareto_model.load_state_dict(torch.load(args.unif_model_load_path))
        duration = 0
        pareto_durs = None
    else:
        print("Running Pareto finetuning...")
        start_time = time.time()
        
        trainer = ModelTrainer(device)
        pareto_model, pareto_durs = trainer.train_pareto_model(
            model=copy.deepcopy(base_model),
            forget_loader=forget_train_loader,
            retain_loader=retain_train_loader,
            retain_val_loader=retain_eval_loader,
            forget_val_loader=forget_eval_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            seed_worker=seed_worker,
            g=g,
            reg=args.with_reg
        )
        
        duration = time.time() - start_time
        print(f"Pareto finetuning duration: {duration:.2f} seconds")
        
        # Save model if path specified
        if args.unif_model_save_path:
            os.makedirs(args.unif_model_save_path, exist_ok=True)
            save_path = os.path.join(args.unif_model_save_path, f"pareto_model_{args.dataset}_{args.model_type}_{args.util_unif_tradeoff_theta}_{args.num_unif}.pth")
            torch.save(pareto_model.state_dict(), save_path)
            print(f"Saved Pareto model to: {save_path}")
    
    return pareto_model, duration, pareto_durs

def run_attack_evaluation(pretrained_model, pareto_model, forget_eval_loader, args, device, num_workers, g, seed_worker):
    """Run attack evaluation if specified"""
    if not args.run_attack:
        return {}, {}, 0
    
    print(f"Running attack evaluation ({args.attack_type}, eps={args.attack_eps})...")
    start_time = time.time()
    
    pretrained_attacked_loader = create_perturbed_dataloader(
        pretrained_model, forget_eval_loader, 
        attack=args.attack_type, epsilon=args.attack_eps,
        num_workers=num_workers, generator=g, worker_init_fn=seed_worker
    )
    
    if pareto_model is not None:
        pareto_attacked_loader = create_perturbed_dataloader(
            pareto_model, forget_eval_loader,
            attack=args.attack_type, epsilon=args.attack_eps,
            num_workers=num_workers, generator=g, worker_init_fn=seed_worker
        )
    else:
        pareto_attacked_loader = pretrained_attacked_loader
    
    duration = time.time() - start_time
    
    pretrained_attacked_metrics = forget_quality(
        pretrained_model, pretrained_attacked_loader,
        args.forget_images_path, "pretrained_attacked", 
        args.batch_size, args.num_classes, args, device
    )
    
    pareto_attacked_metrics = forget_quality(
        pareto_model if pareto_model else pretrained_model, pareto_attacked_loader,
        args.forget_images_path, "pareto_attacked",
        args.batch_size, args.num_classes, args, device
    )
    
    print(f"Attack evaluation duration: {duration:.2f} seconds")
    return pretrained_attacked_metrics, pareto_attacked_metrics, duration

def run_nearest_neighbors_evaluation(pretrained_model, pareto_model, forget_eval_loader, test_loader, args, device):
    """Run nearest neighbors evaluation if enabled"""
    if not args.run_nn:
        return {}, {}
    
    print("Running nearest neighbors evaluation...")
    nn_loader = get_nearest_neighbors_dataloader(forget_eval_loader, test_loader, device)
    
    pretrain_nn_acc = evaluate(pretrained_model, nn_loader, device)
    pretrain_nn_metrics = forget_quality(
        pretrained_model, nn_loader, args.forget_images_path, 
        "pretrained_nn", args.batch_size, args.num_classes, args, device
    )
    
    if pareto_model is not None:
        pareto_nn_acc = evaluate(pareto_model, nn_loader, device)
        pareto_nn_metrics = forget_quality(
            pareto_model, nn_loader, args.forget_images_path,
            "pareto_nn", args.batch_size, args.num_classes, args, device
        )
    else:
        pareto_nn_acc = 0
        pareto_nn_metrics = {'avg_confidence_distance': 0}
    
    print(f"Pretrain NN - Accuracy: {pretrain_nn_acc:.2f}%, Confidence Distance: {pretrain_nn_metrics['avg_confidence_distance']:.4f}")
    if pareto_model is not None:
        print(f"Pareto NN - Accuracy: {pareto_nn_acc:.2f}%, Confidence Distance: {pareto_nn_metrics['avg_confidence_distance']:.4f}")
    
    return {
        'pretrain_acc': pretrain_nn_acc,
        'pretrain_metrics': pretrain_nn_metrics,
        'pareto_acc': pareto_nn_acc,
        'pareto_metrics': pareto_nn_metrics
    }, {}

def run_test_set_sampling(pretrained_model, pareto_model, train_loader, train_eval_loader, forget_train_loader, 
                         forget_eval_loader, test_loader, retain_train_loader, retain_eval_loader, args, device, num_workers, g, seed_worker):
    if not args.run_test_set_sampling:
        return {}, {}
    
    print(f"Running test set sampling (frac={args.frac_of_test})...")
    
    # Create new data splits with test set sampling
    new_train_loader, new_train_eval_loader, new_forget_train_loader, new_forget_eval_loader, new_test_loader = test_sampling(
        args.frac_of_test, train_loader, train_eval_loader, forget_train_loader, 
        forget_eval_loader, test_loader, args, num_workers=num_workers, generator=g, worker_init_fn=seed_worker
    )
    
    # Train new pretrained model or load existing
    if args.new_trained_model_load_path:
        print("Loading new pretrained model from:", args.new_trained_model_load_path)
        initializer = ModelInitializer(device)
        new_pretrained_model = initializer.init_model(args)
        new_pretrained_model.load_state_dict(torch.load(args.new_trained_model_load_path))
    else:
        print("Fine-tuning pretrained model on new training set...")
        trainer = ModelTrainer(device)
        original_epochs = args.epochs
        args.epochs = args.ft_epochs  
        new_pretrained_model = trainer.retrain_from_scratch(
            model=copy.deepcopy(pretrained_model),
            retain_loader=new_train_loader,
            retain_val_loader=new_train_eval_loader,
            args=args, num_workers=num_workers, seed_worker=seed_worker, g=g,
            data_shape=get_data_shape(args), logistic=False
        )
        args.epochs = original_epochs  
    
    # Train new Pareto model or load existing
    if args.new_unif_model_load_path:
        print("Loading new Pareto model from:", args.new_unif_model_load_path)
        initializer = ModelInitializer(device)
        new_pareto_model = initializer.init_model(args)
        new_pareto_model.load_state_dict(torch.load(args.new_unif_model_load_path))
    else:
        print("Training new Pareto model...")
        trainer = ModelTrainer(device)
        original_lr = args.lr
        args.lr = args.new_lr  # Use new learning rate
        new_pareto_model, _ = trainer.train_pareto_model(
            model=copy.deepcopy(new_pretrained_model),
            forget_loader=new_forget_train_loader,
            retain_loader=retain_train_loader,
            retain_val_loader=retain_eval_loader,
            forget_val_loader=new_forget_eval_loader,
            test_loader=new_test_loader,
            device=device,
            args=args,
            seed_worker=seed_worker,
            g=g,
            reg=args.with_reg
        )
        args.lr = original_lr  
    
    # Save new models
    if args.new_trained_model_save_path:
        os.makedirs(args.new_trained_model_save_path, exist_ok=True)
        save_path = os.path.join(args.new_trained_model_save_path, f"new_pretrained_model_{args.dataset}_{args.model_type}.pth")
        torch.save(new_pretrained_model.state_dict(), save_path)
        print(f"Saved new pretrained model to: {save_path}")
    
    if args.new_unif_model_save_path:
        os.makedirs(args.new_unif_model_save_path, exist_ok=True)
        save_path = os.path.join(args.new_unif_model_save_path, f"new_pareto_model_{args.dataset}_{args.model_type}_{args.util_unif_tradeoff_theta}.pth")
        torch.save(new_pareto_model.state_dict(), save_path)
        print(f"Saved new Pareto model to: {save_path}")
    
    # Evaluate new models
    new_pretrained_results = {
        'retain_acc': evaluate(new_pretrained_model, retain_eval_loader, device),
        'test_acc': evaluate(new_pretrained_model, new_test_loader, device),
        'metrics': forget_quality(new_pretrained_model, new_forget_eval_loader, args.forget_images_path, 
                                 "new_pretrained", args.batch_size, args.num_classes, args, device)
    }
    
    new_pareto_results = {
        'retain_acc': evaluate(new_pareto_model, retain_eval_loader, device),
        'test_acc': evaluate(new_pareto_model, new_test_loader, device),
        'metrics': forget_quality(new_pareto_model, new_forget_eval_loader, args.forget_images_path,
                                 "new_pareto", args.batch_size, args.num_classes, args, device)
    }
    
    return new_pretrained_results, new_pareto_results

def evaluate_model_comprehensive(model, model_name, retain_eval_loader, test_loader, forget_eval_loader, args, device):
    """Comprehensive model evaluation"""
    if model is None:
        return None
    
    print(f"Evaluating {model_name}...")
    
    retain_acc = evaluate(model, retain_eval_loader, device)
    test_acc = evaluate(model, test_loader, device)
    forget_acc = evaluate(model, forget_eval_loader, device)
    
    forget_metrics = forget_quality(
        model, forget_eval_loader, args.forget_images_path,
        model_name, args.batch_size, args.num_classes, args, device
    )
    
    results = {
        'retain_acc': retain_acc,
        'test_acc': test_acc,
        'forget_acc': forget_acc,
        'forget_metrics': forget_metrics
    }
    
    print(f"{model_name} - Retain: {retain_acc:.2f}%, Test: {test_acc:.2f}%, Forget: {forget_acc:.2f}%")
    print(f"{model_name} - Uniformity Loss: {forget_metrics['avg_unif_loss']:.4f}, Confidence Distance: {forget_metrics['avg_confidence_distance']:.4f}")
    
    return results

def save_comprehensive_results(all_results, durations, args):
    """Save comprehensive experiment results using the original detailed format"""
    cdt = pytz.timezone("America/Chicago")
    timestamp = datetime.now(cdt).strftime("%m_%d_%H_%M_%S")
    results_filename = f"{args.model_type}_{args.util_unif_tradeoff_theta}_{timestamp}.txt"
    results_path = os.path.join(os.path.dirname(args.results_file_path), results_filename)
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Extract results with proper defaults
    def get_result(results, model_key, metric_key, default=0):
        return results.get(model_key, {}).get(metric_key, default) if results.get(model_key) else default
    
    def get_forget_metric(results, model_key, metric_key, default=0):
        forget_metrics = results.get(model_key, {}).get('forget_metrics', {})
        return forget_metrics.get(metric_key, default) if forget_metrics else default
    
    # Prepare data for the original format
    results_data = {
        "Dataset": args.dataset,
        "Model": args.model_type,
        "Seed": args.seed,
        "Uniformity Loss": args.unif_loss,
        "Theta": args.util_unif_tradeoff_theta,
        "Lambda": args.convex_approx_lambda if args.with_reg else 0,
        "Downsample": args.downsample,
        "Randomize labels": args.randomize_labels,
        "Pretrain learning rate": args.lr,
        "Pretrain epochs": args.epochs,
        "Pareto learning rate": args.pareto_lr,
        "Pareto epochs": args.pareto_epochs,
        "Pretrain opt": args.pretrain_opt,
        "Pareto opt": args.pareto_opt,
        "Pretrain schedule": args.pretrain_schedule,
        "Pareto schedule": args.pareto_schedule,
        "Early stop retain": args.early_stop_retain,
        "Retain acc threshold": args.retain_acc_threshold,
        "Distance threshold": args.distance_threshold,
        "Uniformity Mode": args.uniformity_mode,
        "Frac of test": args.frac_of_test if args.run_test_set_sampling else 0,
        "Confidence distance": args.confidence_distance,
        "Use Schedule": args.use_schedule,
        "Attack epsilon": args.attack_eps if args.run_attack else 0,
        "Attack Type": args.attack_type if args.run_attack else "",
        
        # Model results
        "Pretrained model retain acc": get_result(all_results, 'pretrained', 'retain_acc'),
        "Pretrained model test acc": get_result(all_results, 'pretrained', 'test_acc'),
        "Pretrained model forget acc": get_result(all_results, 'pretrained', 'forget_acc'),
        "Pretrained model avg. uniformity loss over forget set": get_forget_metric(all_results, 'pretrained', 'avg_unif_loss'),
        "Pretrained model avg. max{0,f(x)_t-1/|Y|}": get_forget_metric(all_results, 'pretrained', 'avg_confidence_distance'),
        
        "Warmup model retain acc": get_result(all_results, 'warmup', 'retain_acc'),
        "Warmup model test acc": get_result(all_results, 'warmup', 'test_acc'),
        "Warmup model forget acc": get_result(all_results, 'warmup', 'forget_acc'),
        "Warmup model avg. uniformity loss over forget set": get_forget_metric(all_results, 'warmup', 'avg_unif_loss'),
        "Warmup model avg. max{0, f(x)_t-1/|Y|}": get_forget_metric(all_results, 'warmup', 'avg_confidence_distance'),
        
        "Certified model retain acc": get_result(all_results, 'certified', 'retain_acc'),
        "Certified model test acc": get_result(all_results, 'certified', 'test_acc'),
        "Certified model forget acc": get_result(all_results, 'certified', 'forget_acc'),
        "Certified model avg. uniformity loss over forget set": get_forget_metric(all_results, 'certified', 'avg_unif_loss'),
        "Certified model avg. max{0, f(x)_t-1/|Y|}": get_forget_metric(all_results, 'certified', 'avg_confidence_distance'),
        
        "Retrain model retain acc": get_result(all_results, 'retrain', 'retain_acc'),
        "Retrain model test acc": get_result(all_results, 'retrain', 'test_acc'),
        "Retrain model forget acc": get_result(all_results, 'retrain', 'forget_acc'),
        "Retrain model avg. uniformity loss over forget set": get_forget_metric(all_results, 'retrain', 'avg_unif_loss'),
        "Retrain model avg. max{0, f(x)_t-1/|Y|}": get_forget_metric(all_results, 'retrain', 'avg_confidence_distance'),
        
        "Finetuned Pareto retain acc": get_result(all_results, 'pareto', 'retain_acc'),
        "Finetuned Pareto test acc": get_result(all_results, 'pareto', 'test_acc'),
        "Finetuned Pareto forget acc": get_result(all_results, 'pareto', 'forget_acc'),
        "Finetuned Pareto avg. uniformity loss over forget set": get_forget_metric(all_results, 'pareto', 'avg_unif_loss'),
        "Finetuned Pareto avg. max{0,f(x)_t-1/|Y|}": get_forget_metric(all_results, 'pareto', 'avg_confidence_distance'),
        
        # Attack results
        "Pretrained post-attack avg. max{0,f(x)_t-1/|Y|}": all_results.get('attack', {}).get('pretrained_metrics', {}).get('avg_confidence_distance', 0),
        "Pretrained post-attack forget accuracy": all_results.get('attack', {}).get('pretrained_metrics', {}).get('accuracy', 0),
        "Finetuned post-attack avg. max{0,f(x)_t-1/|Y|}": all_results.get('attack', {}).get('pareto_metrics', {}).get('avg_confidence_distance', 0),
        "Finetuned post-attack forget accuracy": all_results.get('attack', {}).get('pareto_metrics', {}).get('accuracy', 0),
        
        # Nearest neighbors results
        "Pretrain nn accuracy": all_results.get('nn', {}).get('pretrain_acc', 0),
        "Pretrain nn avg. max{0,f(x)_t-1/|Y|}": all_results.get('nn', {}).get('pretrain_metrics', {}).get('avg_confidence_distance', 0),
        "Finetune nn accuracy": all_results.get('nn', {}).get('pareto_acc', 0),
        "Finetune nn avg. max{0,f(x)_t-1/|Y|}": all_results.get('nn', {}).get('pareto_metrics', {}).get('avg_confidence_distance', 0),
        
        # Test set sampling results
        "New pretrained model retain acc": all_results.get('test_sampling', {}).get('new_pretrained', {}).get('retain_acc', 0),
        "New pretrained model test acc": all_results.get('test_sampling', {}).get('new_pretrained', {}).get('test_acc', 0),
        "New pretrained model avg. max{0,f(x)_t-1/|Y|}": all_results.get('test_sampling', {}).get('new_pretrained', {}).get('metrics', {}).get('avg_confidence_distance', 0),
        "New pareto model retain acc": all_results.get('test_sampling', {}).get('new_pareto', {}).get('retain_acc', 0),
        "New pareto model test acc": all_results.get('test_sampling', {}).get('new_pareto', {}).get('test_acc', 0),
        "New pareto model avg. max{0,f(x)_t-1/|Y|}": all_results.get('test_sampling', {}).get('new_pareto', {}).get('metrics', {}).get('avg_confidence_distance', 0),
        
        # Durations
        "Pretrained duration": durations.get('pretrain', 0),
        "Warmup duration": durations.get('warmup', 0),
        "Uniformity duration": durations.get('uniformity', 0),
        "Retrain duration": durations.get('retrain', 0),
        "Pareto finetuning duration": durations.get('pareto', 0),
        "Attack duration": durations.get('attack', 0),
        
        # Pareto timing breakdowns (if available)
        "Retain grad dur prop.": durations.get('pareto_breakdown', {}).get('retain_grad_prop', 0),
        "Forget grad dur prop.": durations.get('pareto_breakdown', {}).get('forget_grad_prop', 0),
        "Projection dur prop.": durations.get('pareto_breakdown', {}).get('projection_prop', 0),
        "Reg dur prop.": durations.get('pareto_breakdown', {}).get('reg_prop', 0),
        "Step dur prop.": durations.get('pareto_breakdown', {}).get('step_prop', 0),
    }
    

    fields = [
        "Dataset", "Model", "Seed", "Uniformity Loss", "Theta", "Lambda", "Downsample", "Randomize labels",
        "Pretrain learning rate", "Pretrain epochs", "Pareto learning rate", "Pareto epochs",
        "Pretrain opt", "Pareto opt", "Pretrain schedule", "Pareto schedule", "Early stop retain",
        "Retain acc threshold", "Distance threshold", "Uniformity Mode", "Frac of test", "Confidence distance",
        "Use Schedule", "Attack epsilon", "Attack Type",
        "Pretrained model retain acc", "Pretrained model test acc", "Pretrained model forget acc",
        "Pretrained model avg. uniformity loss over forget set", "Pretrained model avg. max{0,f(x)_t-1/|Y|}",
        "Warmup model retain acc", "Warmup model test acc", "Warmup model forget acc",
        "Warmup model avg. uniformity loss over forget set", "Warmup model avg. max{0, f(x)_t-1/|Y|}",
        "Certified model retain acc", "Certified model test acc", "Certified model forget acc",
        "Certified model avg. uniformity loss over forget set", "Certified model avg. max{0, f(x)_t-1/|Y|}",
        "Retrain model retain acc", "Retrain model test acc", "Retrain model forget acc",
        "Retrain model avg. uniformity loss over forget set", "Retrain model avg. max{0, f(x)_t-1/|Y|}",
        "Finetuned Pareto retain acc", "Finetuned Pareto test acc", "Finetuned Pareto forget acc",
        "Finetuned Pareto avg. uniformity loss over forget set", "Finetuned Pareto avg. max{0,f(x)_t-1/|Y|}",
        "Pretrained post-attack avg. max{0,f(x)_t-1/|Y|}", "Pretrained post-attack forget accuracy",
        "Finetuned post-attack avg. max{0,f(x)_t-1/|Y|}", "Finetuned post-attack forget accuracy",
        "Pretrain nn accuracy", "Pretrain nn avg. max{0,f(x)_t-1/|Y|}",
        "Finetune nn accuracy", "Finetune nn avg. max{0,f(x)_t-1/|Y|}",
        "New pretrained model retain acc", "New pretrained model test acc", "New pretrained model avg. max{0,f(x)_t-1/|Y|}",
        "New pareto model retain acc", "New pareto model test acc", "New pareto model avg. max{0,f(x)_t-1/|Y|}",
        "Pretrained duration", "Warmup duration", "Uniformity duration", "Retrain duration", 
        "Pareto finetuning duration", "Attack duration", "Retain grad dur prop.", "Forget grad dur prop.",
        "Projection dur prop.", "Reg dur prop.", "Step dur prop."
    ]
    
    with open(results_path, 'w') as f:
        for field in fields:
            value = results_data.get(field, 0)
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            f.write(f"{field}: {formatted_value}\n")
    
    print(f"Comprehensive results saved to: {results_path}")

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def config_to_args(config):
    args = argparse.Namespace()
    
    # Core training parameters
    args.batch_size = config['training']['batch_size']
    args.epochs = config['training']['epochs']
    args.lr = config['training']['lr']
    args.dataset = config['training']['dataset']
    args.model_type = config['training']['model_type']
    args.num_classes = config['training']['num_classes']
    args.seed = config['training']['seed']
    args.downsample = config['training'].get('downsample', False)
    args.randomize_labels = config['training'].get('randomize_labels', False)
    args.train = config['training'].get('train', True)
    args.num_layer = config['training'].get('num_layer', 2)
    args.hidden_dim = config['training'].get('hidden_dim', 100)
    args.pretrain_opt = config['training'].get('pretrain_opt', 'adam')
    args.early_stop_retain = config['training'].get('early_stop_retain', False)
    args.pretrain_schedule = config['training'].get('pretrain_schedule', 'none')
    
    # Experiment control flags
    args.run_pretrain = config['experiment'].get('run_pretrain', True)
    args.run_warmup = config['experiment'].get('run_warmup', False)
    args.run_uniformity = config['experiment'].get('run_uniformity', False)
    args.run_retrain = config['experiment'].get('run_retrain', False)
    args.run_pareto = config['experiment'].get('run_pareto', False)
    args.run_attack = config['experiment'].get('run_attack', False)
    args.run_nn = config['experiment'].get('run_nn', False)
    args.run_test_set_sampling = config['experiment'].get('run_test_set_sampling', False)
    
    # Model paths
    args.trained_model_load_path = config['paths'].get('trained_model_load_path', '')
    args.continue_train_load_path = config['paths'].get('continue_train_load_path', '')
    args.trained_model_save_path = config['paths'].get('trained_model_save_path', '')
    args.unif_model_save_path = config['paths'].get('unif_model_save_path', '')
    args.unif_model_load_path = config['paths'].get('unif_model_load_path', '')
    args.retrain_save_path = config['paths'].get('retrain_save_path', '')
    args.forget_images_path = config['paths']['forget_images_path']
    args.results_file_path = config['paths'].get('results_file_path', './results/experiment_results.txt')
    args.log_file_path = config['paths']['log_file_path']
    
    # Uniformity parameters
    args.num_unif = config.get('uniformity', {}).get('num_unif', 100)
    args.unif_batch_size = config.get('uniformity', {}).get('unif_batch_size', 10)
    args.confidence_distance = config.get('uniformity', {}).get('confidence_distance', 'paper')
    args.uniformity_mode = config.get('uniformity', {}).get('mode', 'exact')
    
    # PGD parameters
    args.C = config.get('pgd', {}).get('C', 1.0)
    args.with_pgd = config.get('pgd', {}).get('with_pgd', False)
    
    # Hessian parameters
    args.concentration_number_b = config.get('hessian', {}).get('concentration_number_b', 0.1)
    args.sample_number_n = config.get('hessian', {}).get('sample_number_n', 100)
    args.convex_approx_lambda = config.get('hessian', {}).get('convex_approx_lambda', 0.01)
    args.min_eig_lambda_min = config.get('hessian', {}).get('min_eig_lambda_min', 0.001)
    args.min_eig_bound_indiv_zeta_min = config.get('hessian', {}).get('min_eig_bound_indiv_zeta_min', 0.001)
    args.scale_Hessian_H = config.get('hessian', {}).get('scale_Hessian_H', 1.0)
    
    # Privacy parameters
    args.privacy_budget_epsilon = config.get('privacy', {}).get('privacy_budget_epsilon', 1.0)
    args.privacy_budget_delta = config.get('privacy', {}).get('privacy_budget_delta', 1e-5)
    
    # Utility-uniformity parameters
    args.compute_theta = config.get('utility_uniformity', {}).get('compute_theta', False)
    args.distance_from_unif = config.get('utility_uniformity', {}).get('distance_from_unif', 0.1)
    args.util_unif_tradeoff_theta = config.get('utility_uniformity', {}).get('util_unif_tradeoff_theta', 1.0)
    args.use_schedule = config.get('utility_uniformity', {}).get('use_schedule', False)
    args.util_unif_init = config.get('utility_uniformity', {}).get('util_unif_init', 1.0)
    args.bound_looseness_rho = config.get('utility_uniformity', {}).get('bound_looseness_rho', 0.1)
    
    # Lipschitz constants
    args.unif_grad_lipschitz_L_K = config.get('lipschitz', {}).get('unif_grad_lipschitz_L_K', 1.0)
    args.retain_grad_lipschitz_L_A = config.get('lipschitz', {}).get('retain_grad_lipschitz_L_A', 1.0)
    args.unif_hessian_lipschitz_M_K = config.get('lipschitz', {}).get('unif_hessian_lipschitz_M_K', 1.0)
    args.retain_hessian_lipschitz_M_A = config.get('lipschitz', {}).get('retain_hessian_lipschitz_M_A', 1.0)
    
    # Additional options
    args.distance_threshold = config.get('options', {}).get('distance_threshold', 0.1)
    args.unif_loss = config.get('options', {}).get('unif_loss', 'square')
    args.with_reg = config.get('options', {}).get('with_reg', False)
    args.warmup = config.get('options', {}).get('warmup', False)
    args.warmup_reg = config.get('options', {}).get('warmup_reg', False)
    args.surgery = config.get('options', {}).get('surgery', False)
    args.retain_acc_threshold = config.get('options', {}).get('retain_acc_threshold', 0.9)
    
    # Pareto parameters
    args.pareto_lr = config.get('pareto', {}).get('lr', args.lr)
    args.pareto_epochs = config.get('pareto', {}).get('epochs', 10)
    args.pareto_opt = config.get('pareto', {}).get('opt', 'adam')
    args.pareto_schedule = config.get('pareto', {}).get('schedule', 'none')
    
    # Attack parameters
    args.attack_eps = config.get('attack', {}).get('eps', 0.1)
    args.attack_type = config.get('attack', {}).get('type', 'fgsm')
    
    # Test set sampling parameters
    args.frac_of_test = config.get('test_set_sampling', {}).get('frac_of_test', 0.1)
    args.new_lr = config.get('test_set_sampling', {}).get('new_lr', args.lr)
    args.new_trained_model_save_path = config.get('test_set_sampling', {}).get('new_trained_model_save_path', '')
    args.new_unif_model_save_path = config.get('test_set_sampling', {}).get('new_unif_model_save_path', '')
    args.new_trained_model_load_path = config.get('test_set_sampling', {}).get('new_trained_model_load_path', '')
    args.new_unif_model_load_path = config.get('test_set_sampling', {}).get('new_unif_model_load_path', '')
    args.ft_epochs = config.get('test_set_sampling', {}).get('ft_epochs', 5)
    
    return args

def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config')
    parser.add_argument('--attack_eps', type=float, default=None, help='Override attack epsilon')
    parser.add_argument('--attack_type', type=str, default=None, help='Override attack type')
    cmd_args = parser.parse_args()
    

    if not os.path.exists(cmd_args.config):
        print(f"Error: Config file {cmd_args.config} not found.")
        exit(1)
    
    config = load_config(cmd_args.config)

    if cmd_args.seed is not None:
        config['training']['seed'] = cmd_args.seed
    if cmd_args.attack_eps is not None:
        config.setdefault('attack', {})['eps'] = cmd_args.attack_eps
    if cmd_args.attack_type is not None:
        config.setdefault('attack', {})['type'] = cmd_args.attack_type
    
    args = config_to_args(config)
    device = setup_environment(args)
    data_shape = get_data_shape(args)
    
    print("log file path: ", args.log_file_path)
    timestamp = datetime.now(cdt).strftime("%m_%d_%H_%M_%S")
    log_filename = f"{args.model_type}_{args.util_unif_tradeoff_theta}_{timestamp}.txt"
    log_file = os.path.join(args.log_file_path, log_filename)  # Use args.log_file_path directly
    print("log file: ", log_file)
    
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log_file = open(log_file, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeStdout(sys.stdout, log_file, print_to_terminal=False)
    sys.stderr = TeeStdout(sys.stderr, log_file, print_to_terminal=False)
    print("PID:", os.getpid())
    
 
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    num_workers = 2
    
 
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_type}")
    print(f"Seed: {args.seed}")
    enabled_experiments = []
    if args.run_pretrain: enabled_experiments.append("Pretrain")
    if args.run_warmup: enabled_experiments.append("Warmup")
    if args.run_uniformity: enabled_experiments.append("Uniformity")
    if args.run_retrain: enabled_experiments.append("Retrain")
    if args.run_pareto: enabled_experiments.append("Pareto")
    if args.run_attack: enabled_experiments.append("Attack")
    if args.run_nn: enabled_experiments.append("NearestNeighbors")
    if args.run_test_set_sampling: enabled_experiments.append("TestSetSampling")
    print(f"Enabled experiments: {', '.join(enabled_experiments)}")
    print("=" * 80)
    
    try:
        pretrained_model = None
        warmup_model = None
        certified_model = None
        retrain_model = None
        pareto_model = None
        # Load data and create splits
        print("Loading data...")
        train_loader, test_loader, train_eval_loader, train_dataset, test_dataset = load_data(
            args, device, num_workers=num_workers, generator=g, worker_init_fn=seed_worker
        )
        print("Transform:", train_dataset.transform)
        print("Dataset length:", len(train_dataset))
        xb, yb = next(iter(train_loader))
        print("First 5 labels:", yb[:5].tolist())
        print("First image checksum:", float(xb[0].sum()))
        
        retain_train_loader, retain_eval_loader, forget_train_loader, forget_eval_loader = data_splitter(
            train_loader, train_eval_loader, args, num_workers=num_workers, generator=g, worker_init_fn=seed_worker
        )
        
        print(f"Data loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        print(f"Retain: {len(retain_train_loader.dataset)}, Forget: {len(forget_train_loader.dataset)}")

        models = {}
        all_results = {}
        durations = {}
        
        # 1: pretrained Model
        if args.run_pretrain:
            print("\n" + "=" * 60)
            print("STAGE 1: PRETRAINING")
            print("=" * 60)
            pretrained_model, pretrain_dur = load_or_train_pretrained_model(
                args, device, train_loader, train_eval_loader, test_loader, data_shape, num_workers, seed_worker, g
            )
            models['pretrained'] = pretrained_model
            durations['pretrain'] = pretrain_dur
            all_results['pretrained'] = evaluate_model_comprehensive(
                pretrained_model, 'pretrained', retain_eval_loader, test_loader, forget_eval_loader, args, device
            )
        else:
            initializer = ModelInitializer(device)
            if not args.run_retrain:
                pretrained_model = initializer.init_model(args)
                pretrained_model.load_state_dict(torch.load(args.trained_model_load_path))
                print("Skipping pretraining stage")
        
        # 2: warmup (optional)
        if args.run_warmup and pretrained_model is not None:
            print("\n" + "=" * 60)
            print("STAGE 2: WARMUP TRAINING")
            print("=" * 60)
            warmup_model, warmup_dur = run_warmup_training(
                pretrained_model, forget_train_loader, retain_train_loader, retain_eval_loader, 
                forget_eval_loader, test_loader, args, device, seed_worker, g
            )
            models['warmup'] = warmup_model
            durations['warmup'] = warmup_dur
            all_results['warmup'] = evaluate_model_comprehensive(
                warmup_model, 'warmup', retain_eval_loader, test_loader, forget_eval_loader, args, device
            )
           
            base_model = warmup_model
        else:
            # After the warmup stage:
            if warmup_model is not None:
                base_model = warmup_model
            elif pretrained_model is not None:
                base_model = pretrained_model  
            else:
                base_model = None
            print("Skipping warmup stage")
        
        # 3: uniformity certification
        if args.run_uniformity and base_model is not None:
            print("\n" + "=" * 60)
            print("STAGE 3: UNIFORMITY CERTIFICATION")
            print("=" * 60)
            certified_model, uniformity_dur = run_uniformity_certification(
                base_model, retain_eval_loader, forget_eval_loader, args, device
            )
            models['certified'] = certified_model
            durations['uniformity'] = uniformity_dur
            all_results['certified'] = evaluate_model_comprehensive(
                certified_model, 'certified', retain_eval_loader, test_loader, forget_eval_loader, args, device
            )
        else:
            certified_model = None
            durations['uniformity'] = 0
            print("Skipping uniformity certification stage")
        
        # 4: retrain baseline
        if args.run_retrain:
            print("\n" + "=" * 60)
            print("STAGE 4: RETRAIN BASELINE")
            print("=" * 60)
            retrain_model, retrain_dur = run_retrain_baseline(
                args, device, retain_train_loader, retain_eval_loader, test_loader, data_shape, num_workers, seed_worker, g
            )
            models['retrain'] = retrain_model
            durations['retrain'] = retrain_dur
            all_results['retrain'] = evaluate_model_comprehensive(
                retrain_model, 'retrain', retain_eval_loader, test_loader, forget_eval_loader, args, device
            )
        else:
            retrain_model = None
            durations['retrain'] = 0
            print("Skipping retrain baseline stage")
        
        # 5: pareto Finetuning
        if args.run_pareto and base_model is not None:
            print("\n" + "=" * 60)
            print("STAGE 5: PARETO FINETUNING")
            print("=" * 60)
            pareto_model, pareto_dur, pareto_durs = run_pareto_finetuning(
                base_model, forget_train_loader, retain_train_loader, retain_eval_loader, 
                forget_eval_loader, test_loader, args, device, seed_worker, g
            )
            models['pareto'] = pareto_model
            durations['pareto'] = pareto_dur
            if pareto_durs:
                total_pareto_time = sum(pareto_durs.values())
                durations['pareto_breakdown'] = {
                    'retain_grad_prop': pareto_durs.get('retain_grad_dur', 0) / total_pareto_time,
                    'forget_grad_prop': pareto_durs.get('forget_grad_dur', 0) / total_pareto_time,
                    'projection_prop': pareto_durs.get('projection_dur', 0) / total_pareto_time,
                    'reg_prop': pareto_durs.get('reg_dur', 0) / total_pareto_time,
                    'step_prop': pareto_durs.get('step_dur', 0) / total_pareto_time
                }
            all_results['pareto'] = evaluate_model_comprehensive(
                pareto_model, 'pareto', retain_eval_loader, test_loader, forget_eval_loader, args, device
            )
        else:
            pareto_model = None
            durations['pareto'] = 0
            print("Skipping Pareto finetuning stage")
        
        # 6: attack evaluation
        if args.run_attack and pretrained_model is not None:
            print("\n" + "=" * 60)
            print("STAGE 6: ATTACK EVALUATION")
            print("=" * 60)
            pretrained_attacked_metrics, pareto_attacked_metrics, attack_dur = run_attack_evaluation(
                pretrained_model, pareto_model, forget_eval_loader, args, device, num_workers, g, seed_worker
            )
            durations['attack'] = attack_dur
            all_results['attack'] = {
                'pretrained_metrics': pretrained_attacked_metrics,
                'pareto_metrics': pareto_attacked_metrics
            }
            print(f"Pretrained post-attack confidence distance: {pretrained_attacked_metrics.get('avg_confidence_distance', 0):.4f}")
            if pareto_model:
                print(f"Pareto post-attack confidence distance: {pareto_attacked_metrics.get('avg_confidence_distance', 0):.4f}")
        else:
            durations['attack'] = 0
            print("Skipping attack evaluation stage")
        
        # 7: nearest neighbors
        if args.run_nn and pretrained_model is not None:
            print("\n" + "=" * 60)
            print("STAGE 7: NEAREST NEIGHBORS EVALUATION")
            print("=" * 60)
            nn_results, _ = run_nearest_neighbors_evaluation(
                pretrained_model, pareto_model, forget_eval_loader, test_loader, args, device
            )
            all_results['nn'] = nn_results
        else:
            print("Skipping nearest neighbors evaluation stage")
        
        # 8: Test Set sampling
        if args.run_test_set_sampling and pretrained_model is not None:
            print("\n" + "=" * 60)
            print("STAGE 8: TEST SET SAMPLING")
            print("=" * 60)
            new_pretrained_results, new_pareto_results = run_test_set_sampling(
                pretrained_model, pareto_model, train_loader, train_eval_loader, forget_train_loader,
                forget_eval_loader, test_loader, retain_train_loader, retain_eval_loader, args, device, num_workers, g, seed_worker
            )
            all_results['test_sampling'] = {
                'new_pretrained': new_pretrained_results,
                'new_pareto': new_pareto_results
            }
            print(f"New pretrained - Retain: {new_pretrained_results.get('retain_acc', 0):.2f}%, Test: {new_pretrained_results.get('test_acc', 0):.2f}%")
            print(f"New Pareto - Retain: {new_pareto_results.get('retain_acc', 0):.2f}%, Test: {new_pareto_results.get('test_acc', 0):.2f}%")
        else:
            print("Skipping test set sampling stage")
        
        # summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        total_time = sum(v for v in durations.values() if isinstance(v, (int, float)))
        print(f"Total experiment time: {total_time:.2f} seconds")
        
        for stage, duration in durations.items():
            print(stage,": ", duration)
            if stage != 'pareto_breakdown' and isinstance(duration, (int, float)) and duration > 0:
                print(f"  {stage.capitalize()}: {duration:.2f} seconds")
        
        if 'pretrained' in all_results:
            print(f"\nKey Results:")
            pretrained = all_results['pretrained']
            print(f"  Pretrained - Retain: {pretrained['retain_acc']:.2f}%, Forget: {pretrained['forget_acc']:.2f}%")
            
            if 'pareto' in all_results:
                pareto = all_results['pareto']
                print(f"  Pareto - Retain: {pareto['retain_acc']:.2f}%, Forget: {pareto['forget_acc']:.2f}%")
            
            if 'retrain' in all_results:
                retrain = all_results['retrain']
                print(f"  Retrain - Retain: {retrain['retain_acc']:.2f}%, Forget: {retrain['forget_acc']:.2f}%")

        save_comprehensive_results(all_results, durations, args)
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print("Log file closed.")

if __name__ == "__main__":
    cdt = pytz.timezone("America/Chicago")
    main()