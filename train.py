import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import time
from tqdm import tqdm
import torch.nn.functional as F
import gc
import random
from typing import List
import sys
import os 
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


from models import ModelInitializer
from evaluator import evaluate, distance_from_uniform
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def unflatten_from_flat(flat_grad: torch.Tensor, optimizer) -> List[torch.Tensor]:
    """
    Splits a 1-D flat_grad back into one tensor per parameter,
    using each param’s own .shape (and .numel()) to know how big a chunk to take.
    """
    grads, idx = [], 0
    for group in optimizer.param_groups:
        for p in group['params']:
            numel = p.numel()
            chunk = flat_grad[idx: idx + numel].view_as(p).clone()
            grads.append(chunk)
            idx += numel
    assert idx == flat_grad.numel(), "flat_grad size mismatch!"
    return grads


class ModelTrainer:
    def __init__(self, device):
        self.device = device

    def retrain_from_scratch(self, model,
                             retain_loader, retain_val_loader, 
                             args, num_workers=0, seed_worker=None, g=None, data_shape=None, 
                             logistic = False, test_loader = None):
        """
        Retrain the given model using only the retain dataset (without forget set).
    
        """
        # for name, module in model.named_modules():
        #         print(name, module)
        # After model creation
        # print("Model state dict keys:", list(model.state_dict().keys())[:5])
        # print("First layer weight sum:", model.conv1.weight.sum().item())
        # print("Linear layer weight sum:", model.linear.weight.sum().item())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mixup_fn = Mixup(
            mixup_alpha=0.2,      # tuneable
            cutmix_alpha=1.0,     # keep CutMix active
            prob=1.0,             # always apply (can try 0.5–0.8)
            switch_prob=0.5,      # switch between Mixup and CutMix
            mode='batch',
            label_smoothing=0.1,  # aligns with your smoothing
            num_classes=args.num_classes
        )

        
        retrain_model = model
        retrain_model.to(self.device)
        
        if args.dataset == "CIFAR100": 
            smoothing = 0.1
        else: 
            smoothing = 0

        #criterion = nn.CrossEntropyLoss(label_smoothing = smoothing)
        criterion = SoftTargetCrossEntropy()
        
        if args.pretrain_opt == "SGD": 
            # L2 regularization if reg = True
            if args.convex_approx_lambda != 0:
                print("optimizer: SGD with reg")
                optimizer = optim.SGD(retrain_model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay = args.convex_approx_lambda)
            else:
                print("optimizer: SGD without reg")
                optimizer = optim.SGD(retrain_model.parameters(), lr=args.lr,
                        momentum=0.9)
        elif args.pretrain_opt == "ADAM": 
            # L2 regularization if reg = True
            if args.convex_approx_lambda != 0:
                print("optimizer: ADAM with reg")
                optimizer = optim.AdamW(retrain_model.parameters(), lr=args.lr, 
                        weight_decay=args.convex_approx_lambda)
            else:
                print("optimizer: ADAM without reg")
                optimizer = optim.Adam(retrain_model.parameters(), lr=args.lr)
        else: 
            raise("Optimizer not supported")

        if args.pretrain_schedule: 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs
            )
                
        train_loader = retain_loader
        train_dataset = train_loader.dataset
        
        train_iter = iter(train_loader)
        first_batch_x, first_batch_y = next(train_iter)
        print(f"First batch shape: {first_batch_x.shape}")
        print(f"First batch labels: {first_batch_y[:30]}")  # First 10 labels
        print(f"First sample mean: {first_batch_x[0].mean():.6f}")
        print(f"First sample std: {first_batch_x[0].std():.6f}")
        
        # Training loop
        retrain_model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            
            with tqdm(train_loader, unit="batch", file=sys.__stdout__) as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if mixup_fn is not None:
                        inputs, labels = mixup_fn(inputs, labels)
                    #print("Labels min:", labels.min().item(), "max:", labels.max().item())
                    
                    optimizer.zero_grad()
                    
                    outputs = retrain_model(inputs)
                    #print("Model output shape:", outputs.shape)
                    loss = criterion(outputs, labels)
                    if torch.isnan(loss):
                        print("LOSS IS NAN!")
                        print("Logits min/max:", outputs.min().item(), outputs.max().item())
                        print("Weights checksum:", sum(p.sum().item() for p in model.parameters()))
                        break
                    loss.backward()
                    optimizer.step()
                    
                    # apply PGD if args.with_pgd is true
                    if hasattr(args, 'with_pgd') and args.with_pgd:
                        self._apply_pgd(retrain_model, args.C)
                    
                    running_loss += loss.item() * inputs.size(0)
                    tepoch.set_postfix(loss=loss.item())
            
            # Calculate loss
            epoch_loss = running_loss / len(train_dataset)
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

            if args.pretrain_schedule: 
                scheduler.step()

            if args.early_stop_retain > 0: 
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_X, val_y in retain_val_loader:
                        val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                        outputs = model(val_X) 
                        _, predicted = torch.max(outputs.data, 1)
                        total += val_y.size(0)
                        correct += (predicted == val_y).sum().item()
                        
                        del outputs, predicted
                    torch.cuda.empty_cache()
                        
                    retain_accuracy = 100 * correct / total
                    print(f"Current retain accuracy: {retain_accuracy}")
                    if retain_accuracy > args.early_stop_retain: break 
        
        return retrain_model
    
    
    def train_pareto_model(self, model,
                           forget_loader, retain_loader, retain_val_loader, forget_val_loader, test_loader,
                           device='cuda', args=None, seed_worker = None, g =None, reg = False):

        gc.collect()
        torch.cuda.empty_cache()
        
        model = model.to(device)
        best_model_state = None
        best_distance = float('inf') 

        lambda_reg = args.convex_approx_lambda
        lr = args.pareto_lr 
        num_epochs = args.pareto_epochs
        batch_size = max(4, args.batch_size // 4)  
        C = args.C

        if args.pareto_opt == "SGD": 
            optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.9,weight_decay=0.0005)
        elif args.pareto_opt == "ADAM": 
            optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0)
        else: 
            raise("Optimizer not supported")

        if args.pareto_schedule: 
            def lr_schedule(epoch):
                if epoch < 30: 
                    return 1.0             
                else: 
                    return 0.2             
           
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)


        ce_criterion = nn.CrossEntropyLoss()
        if args.unif_loss == "kl_forward" or args.unif_loss == "kl_reverse": 
            unif_criterion = nn.KLDivLoss(reduction='batchmean')
        if args.unif_loss == "square": 
            unif_criterion = torch.nn.MSELoss(reduction='mean')

        forget_grad_durs = []
        retain_grad_durs = []
        projection_durs = []
        reg_durs = []
        step_durs = []
        for epoch in range(num_epochs):
            model.train()


            if args.use_schedule and epoch >= 10 and theta < args.util_unif_tradeoff_theta: 
                theta = theta + 0.1

            # after warmup_epochs epochs, start dropping theta considerably so model accuracy can shoot back up
            warmup_epochs = 20
            if args.use_schedule: 
                if epoch <= warmup_epochs:
                    theta = args.util_unif_init
                    print(f"Theta: {theta}")
                else:
                    t = epoch - warmup_epochs
                    total = args.epochs - warmup_epochs
                    theta = args.util_unif_init + (args.util_unif_tradeoff_theta - args.util_unif_init) * (t / total)               

                    print(f"Theta: {theta}")
            else: 
                theta = args.util_unif_tradeoff_theta

            print(f"Epoch {epoch+1}: Computing and applying gradients")
            unif_losses, ce_losses = [], []
            flat_unif_grads, flat_ce_grads = [], []

            print("Building forget gradient")
            forget_grad_start = time.time()
            if args.surgery: 

                if args.dataset == "KMNIST" or args. dataset == "MNIST": 
                    temperature = 2.0
                else:
                    temperature = 1.0

                # 1a) forget‐loader
                g_u = 0 
                for forget_X, forget_y in forget_loader:
                    forget_X, forget_y = forget_X.to(device), forget_y.to(device)
                

                    outputs = model(forget_X)  / temperature
                    softmax_outputs = F.softmax(outputs, dim=1)
                    log_softmax_outputs = F.log_softmax(outputs, dim=1)
                    
                    num_classes = outputs.size(1)
                    log_uniform = torch.full_like(outputs, -torch.log(torch.tensor(num_classes, dtype=outputs.dtype)))
                    uniform = torch.full_like(softmax_outputs, 1.0/num_classes)

                    if args.unif_loss == "kl_forward":
                        loss_u = unif_criterion(log_uniform, softmax_outputs + 1e-8) / len(forget_loader) # computes KL(softmax_outputs, unif)
                    if args.unif_loss == "kl_reverse":
                        loss_u = unif_criterion(log_softmax_outputs, uniform) / len(forget_loader) # computes KL(unif, outputs)
                    if args.unif_loss == "square":
                        loss_u = unif_criterion(softmax_outputs, uniform) * 0.5 * args.num_classes # rescaling

                    # scale
                    loss_u = theta * loss_u
                    unif_losses.append(loss_u.item())

                    # get gradient and add 
                    grads = torch.autograd.grad(loss_u, model.parameters(),
                                        allow_unused=True)
                    flat_unif_grad = parameters_to_vector([
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(grads, model.parameters())
                    ]).to(device).detach()

                    g_u += flat_unif_grad

                    # free everything else on CUDA
                    optimizer.zero_grad(set_to_none=True)
                    del outputs, softmax_outputs, log_softmax_outputs, uniform, log_uniform, loss_u, grads, flat_unif_grad
                    torch.cuda.empty_cache()

                forget_grad_dur = time.time() - forget_grad_start
                #print(f"Forget grad dur: {forget_grad_dur}")
                forget_grad_durs.append(forget_grad_dur)

                #print("Building retain gradient")
                retain_grad_start = time.time()
                # 1b) retain‐loader (exact same pattern)
                g_c = 0
                for X, y in retain_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    loss_c = ce_criterion(outputs, y)  * (1 - theta) / len(retain_loader)
                    ce_losses.append(loss_c.item())

                    # get grads and add 
                    grads = torch.autograd.grad(loss_c, model.parameters(),
                                                allow_unused=True)

                    flat_ce_grad = parameters_to_vector([
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(grads, model.parameters())
                    ]).to(device).detach()
                
                    g_c += flat_ce_grad

                    # free everything on CUDA
                    optimizer.zero_grad(set_to_none=True)
                    del outputs, loss_c, grads, flat_ce_grad
                    torch.cuda.empty_cache()
                
                retain_grad_dur = time.time() - retain_grad_start
                #print(f"Retain grad dur: {retain_grad_dur}")
                retain_grad_durs.append(retain_grad_dur)

                # detach so no leftover graph
                g_u, g_c = g_u.detach(), g_c.detach()
                norm_g_u = g_u.norm()
                norm_g_c = g_c.norm()
                print(f"g_u_norm: {norm_g_u}")
                print(f"g_c_norm: {norm_g_c}")
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                print(f"Cosine sim: {cos(g_u, g_c)}")

                # call projection (on GPU, now, in matrix form) for gradient surgery
                projection_start = time.time()
                dot_uc = (g_u * g_c).sum()
                if torch.rand(1) < 0.5:
                    # project u onto c
                    if dot_uc < 0:
                        print("projecting unif grad onto ce grad first")
                        g_u_proj = g_u - (dot_uc / g_c.pow(2).sum()) * g_c
                        g_c_proj = g_c - (dot_uc / g_u.pow(2).sum()) * g_u
                    else: 
                        g_u_proj = g_u
                        g_c_proj = g_c
                else:
                    # project c onto u
                    if dot_uc < 0:
                        print("projecting ce grad onto unif grad first")
                        g_c_proj = g_c - (dot_uc / g_u.pow(2).sum()) * g_u
                        g_u_proj = g_u - (dot_uc / g_c.pow(2).sum()) * g_c
                    else: 
                        g_u_proj = g_u
                        g_c_proj = g_c


                merged = g_u_proj + g_c_proj

                # put grads back into model params before step
                for p, g_chunk in zip(
                        [p for g in optimizer.param_groups for p in g['params']],
                        unflatten_from_flat(merged, optimizer)
                    ):
                    p.grad = g_chunk.to(p.device)

                projection_dur = time.time() - projection_start
                #print(f"Projection dur: {projection_dur}")
                projection_durs.append(projection_dur)

                # add reg appropriately 
                reg_start = time.time()
                if args.with_reg: 
                    for p in optimizer.param_groups[0]['params']:
                        if p.requires_grad:
                            p.grad.data.add_(lambda_reg, p.data)
                reg_dur = time.time() - reg_start
                #print(f"Reg grad dur: {reg_dur}")
                reg_durs.append(reg_dur)

                # now step 
                optimizer_step_start = time.time()
                optimizer.step()
                step_dur = time.time() - optimizer_step_start
                #print(f"Step dur: {step_dur}")
                step_durs.append(step_dur)

                if args.pareto_schedule: 
                    scheduler.step()

            else: # not recording times here, it's okay
                optimizer.zero_grad()
                
                unif_losses = []
                for forget_X, forget_y in forget_loader:
                    forget_X, forget_y = forget_X.to(device), forget_y.to(device)
                    
                    if args.dataset == "FMNIST" or args.dataset == "KMNIST" or args.dataset == "MNIST": 
                        temperature = 2.0 # trying 
                    else:
                        temperature = 1.0
                    outputs = model(forget_X)  / temperature
                    softmax_outputs = F.softmax(outputs, dim=1)
                    log_softmax_outputs = F.log_softmax(outputs, dim=1)
                    
                    num_classes = outputs.size(1)
                    log_uniform = torch.full_like(outputs, -torch.log(torch.tensor(num_classes, dtype=outputs.dtype)))
                    uniform = torch.full_like(softmax_outputs, 1.0/num_classes)
                    
                    if args.unif_loss == "kl_forward":
                        unif_loss = unif_criterion(log_uniform, softmax_outputs + 1e-8) / len(forget_loader) # computes KL(softmax_outputs, unif)
                    if args.unif_loss == "kl_reverse":
                        unif_loss = unif_criterion(log_softmax_outputs, uniform) / len(forget_loader) # computes KL(unif, outputs)
                    if args.unif_loss == "square":
                        unif_loss = unif_criterion(softmax_outputs, uniform) * 0.5 * args.num_classes # rescaling
                    
                    scaled_unif_loss = theta * unif_loss
                    scaled_unif_loss.backward()
                    unif_losses.append(scaled_unif_loss.item())
            
                    del outputs, softmax_outputs, log_softmax_outputs, log_uniform, uniform, unif_loss, scaled_unif_loss
                    torch.cuda.empty_cache()
                
                scaled_ce_loss = 0
                ce_losses = []

                for retain_X, retain_y in retain_loader:
                    retain_X, retain_y = retain_X.to(device), retain_y.to(device)
                    
                    outputs = model(retain_X)
                    ce_loss = ce_criterion(outputs, retain_y) / len(retain_loader)
                    scaled_ce_loss = (1 - theta) * ce_loss
                    scaled_ce_loss.backward()
                    ce_losses.append(scaled_ce_loss.item())
                                    
                    del outputs, ce_loss, scaled_ce_loss
                    torch.cuda.empty_cache()

                # add regularization
                if reg: 
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.add_(lambda_reg * param.data)
                
                optimizer.step()

            # pgd
            if hasattr(args, 'with_pgd') and args.with_pgd:
                self._apply_pgd(model, C)

            # get logging
            avg_unif_loss = sum(unif_losses) / len(unif_losses)
            avg_ce_loss   = sum(ce_losses)   / len(ce_losses)
                    
            #calculate accuracy on retain set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for val_X, val_y in retain_val_loader:
                    val_X, val_y = val_X.to(device), val_y.to(device)
                    outputs = model(val_X) 
                    _, predicted = torch.max(outputs.data, 1)
                    total += val_y.size(0)
                    correct += (predicted == val_y).sum().item()
                    
                    del outputs, predicted
                    torch.cuda.empty_cache()
                    
            retain_accuracy = 100 * correct / total

            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for test_X, test_y in test_loader:
                    test_X, test_y = test_X.to(device), test_y.to(device)
                    outputs = model(test_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_y.size(0)
                    correct += (predicted == test_y).sum().item()

                    del outputs, predicted
                    torch.cuda.empty_cache()
            test_accuracy = 100 * correct / total
            
            distances = []
            model.eval()
            with torch.no_grad():
                for val_X, val_y in forget_val_loader:
                    val_X = val_X.to(device)
                    outputs = model(val_X)
                    softmax_outputs = F.softmax(outputs, dim=1)
                    num_classes = softmax_outputs.size(1)
                    distance = distance_from_uniform(softmax_outputs + 1e-8, num_classes, args)
                    distances.append(distance.item())
                    del outputs, softmax_outputs
                    torch.cuda.empty_cache()
            avg_distance_from_uniform = sum(distances) / len(distances) if distances else 0

            distances = []
            model.eval()
            with torch.no_grad():
                for val_X, val_y in retain_val_loader:
                    val_X = val_X.to(device)
                    outputs = model(val_X)
                    softmax_outputs = F.softmax(outputs, dim=1)
                    num_classes = softmax_outputs.size(1)
                    distance = distance_from_uniform(softmax_outputs + 1e-8, num_classes, args)
                    distances.append(distance.item())
                    del outputs, softmax_outputs
                    torch.cuda.empty_cache()
            avg_retain_dist = sum(distances) / len(distances) if distances else 0

            distances = []
            model.eval()
            with torch.no_grad():
                for val_X, val_y in test_loader:
                    val_X = val_X.to(device)
                    outputs = model(val_X)
                    softmax_outputs = F.softmax(outputs, dim=1)
                    num_classes = softmax_outputs.size(1)
                    distance = distance_from_uniform(softmax_outputs + 1e-8, num_classes, args)
                    distances.append(distance.item())
                    del outputs, softmax_outputs
                    torch.cuda.empty_cache()
            avg_test_dist = sum(distances) / len(distances) if distances else 0
            
            print(f"Epoch {epoch+1}: Avg Distance from Uniform (Forget Set): {avg_distance_from_uniform:.6f}, Avg Distance from Uniform (Retain Set): {avg_retain_dist:.6f}, Avg  Distance from Uniform (Test set): {avg_test_dist:.6f}, Avg CE Loss: {avg_ce_loss:.4f}, Avg Unif Loss: {avg_unif_loss:.4f}, Retain Accuracy: {retain_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        
            retain_acc_threshold = args.retain_acc_threshold
            distance_threshold = args.distance_threshold
            
            print(f"retain acc threshold: {retain_acc_threshold}")
            print(f"distance threshold: {distance_threshold}")
            if avg_distance_from_uniform < distance_threshold and retain_accuracy >= retain_acc_threshold:
                if avg_distance_from_uniform < best_distance:
                    best_distance = avg_distance_from_uniform
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(best_model_state)
                    print(f"Saved new best model at epoch {epoch+1} with dist={avg_distance_from_uniform:.4f}, retain acc={retain_accuracy:.2f}")
                
            else:
                print("No model satisfied early stopping conditions.")

            gc.collect()
            torch.cuda.empty_cache()

        if args.surgery: 
            durs = { 
                "forget_grad_dur": sum(forget_grad_durs) / len(forget_grad_durs),
                "retain_grad_dur": sum(retain_grad_durs) / len(retain_grad_durs), 
                "projection_dur": sum(projection_durs) / len(projection_durs),
                "reg_dur": sum(reg_durs) / len(reg_durs),
                "step_dur": sum(step_durs) / len(step_durs) 
            }
        else: 
            durs = {}

        if best_model_state is None: # if we never reach early stopping conditions, still save safely
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} 
            best_distance = avg_distance_from_uniform

        model.load_state_dict(best_model_state) # return the best model
        return model, durs


    def _apply_pgd(self, model, C):
        """
        Apply PGD to ensure model parameters satisfy ||w||_2 ≤ C.
        
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param_norm = torch.norm(param)
                    if param_norm > C:
                        param.mul_(C / param_norm)
                    
    
    
    def validate_data(self, X, y, dataset_name="Dataset"):
        """
        Validate that the data is not corrupted
        """
        print(f"\n--- Validating {dataset_name} ---")
    
        print(f"Contains NaN: {torch.isnan(X).any().item()}")
        zero_percent = (X == 0).float().mean().item() * 100
        print(f"Percentage of zeros: {zero_percent:.2f}%")
        
 
        all_same = (X == X[0,0]).all().item()
        print(f"All values identical: {all_same}")
        
        
        # Check label distribution
        unique_labels, counts = torch.unique(y, return_counts=True)
        print(f"Number of classes: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label.item()}: {count.item()} samples ({100*count.item()/len(y):.2f}%)")
        
 
        
        # Check a few random samples
        print("\nSample data points:")
        indices = torch.randint(0, len(X), (3,))
        for idx in indices:
            print(f"Sample {idx}:")
            print(f"  Label: {y[idx].item()}")
            print(f"  Data stats - Min: {X[idx].min().item():.4f}, Max: {X[idx].max().item():.4f}, Mean: {X[idx].mean().item():.4f}")
        
        return not (torch.isnan(X).any().item() or all_same)
    