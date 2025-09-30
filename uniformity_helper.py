import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Subset
import time
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import copy
import gc

# Function to compute gradient of M_theta(D) exactly
def exact_grad(forget_set_loader, retain_set_loader, pretrained_model, args, device, theta): 
    print("getting gradient")
    # get original named parameters
    orig_named = [(n, p) for n, p in pretrained_model.named_parameters() 
                  if p.requires_grad]
    orig_names = [n for n, _ in orig_named]

    # copy model, zero out grads
    lambda_reg = args.convex_approx_lambda
    model_copy = copy.deepcopy(pretrained_model).to(device)
    model_copy.zero_grad()
    model_copy.train()

    # get copied named parameters
    copy_named  = [(n, p) for n, p in model_copy.named_parameters() 
                   if p.requires_grad]
    copy_names  = [n for n, _ in copy_named]
    params      = [p for _, p in copy_named]

    # check that nothing is out of order
    assert orig_names == copy_names, (
        "Parameter order mismatch!\n"
        f"orig: {orig_names}\n"
        f"copy: {copy_names}"
    )

    # initialize criterions
    ce_criterion = nn.CrossEntropyLoss() # L_A
    # L_K
    if args.unif_loss == "kl_forward" or args.unif_loss == "kl_reverse": 
        unif_criterion = nn.KLDivLoss(reduction='batchmean')
    if args.unif_loss == "square": 
        unif_criterion = torch.nn.MSELoss(reduction='mean')
    
    # L_K grad
    forget_processed = 0
    for forget_X, forget_y in forget_set_loader:
        forget_X, forget_y = forget_X.to(device), forget_y.to(device)
        
        outputs = model_copy(forget_X)
        softmax_outputs = F.softmax(outputs, dim=1)
        log_softmax_outputs = F.log_softmax(outputs, dim=1)
        
        num_classes = outputs.size(1)
        log_uniform = torch.full_like(outputs, -torch.log(torch.tensor(num_classes, dtype=outputs.dtype)))
        uniform = torch.full_like(softmax_outputs, 1.0/num_classes)
        
        if args.unif_loss == "kl_forward":
            unif_loss = unif_criterion(log_uniform, softmax_outputs) / len(forget_set_loader) # computes KL(softmax_outputs, unif)
        if args.unif_loss == "kl_reverse":
            unif_loss = unif_criterion(log_softmax_outputs, uniform) / len(forget_set_loader) # computes KL(unif, outputs)
        if args.unif_loss == "square":
            unif_loss = unif_criterion(softmax_outputs, uniform) * 0.5 * args.num_classes # rescaling 
        
        scaled_unif_loss = theta * unif_loss
        scaled_unif_loss.backward()
        forget_processed += 1

        del outputs, softmax_outputs, log_softmax_outputs, log_uniform, uniform, unif_loss, scaled_unif_loss
        torch.cuda.empty_cache()
    
    # L_A grad
    retain_processed = 0
    for retain_X, retain_y in retain_set_loader:
        retain_X, retain_y = retain_X.to(device), retain_y.to(device)
        
        outputs = model_copy(retain_X)
        ce_loss = ce_criterion(outputs, retain_y) / len(retain_set_loader)
        scaled_ce_loss = (1 - theta) * ce_loss
        scaled_ce_loss.backward()
        retain_processed += 1
                            
        del outputs, ce_loss, scaled_ce_loss
        torch.cuda.empty_cache()
    
    # reg grad
    if args.with_reg: 
        for param in model_copy.parameters():
            if param.grad is not None:
                param.grad.add_(lambda_reg * param.data)

            
    # return gradient
    grads = []
    for name, p in copy_named:
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grads.append(g.clone())
    return grads


# Function to quickly compute gradient of objective of M_\theta(D) by only using D_f
def fast_grad(forget_set_loader, forget_set, retain_set, pretrained_model, args, device, theta):
    # get original named parameters
    orig_named = [(n, p) for n, p in pretrained_model.named_parameters() 
                  if p.requires_grad]
    orig_names = [n for n, _ in orig_named]

    # copy model, zero out grads
    lambda_reg = args.convex_approx_lambda
    model_copy = copy.deepcopy(pretrained_model).to(device)
    model_copy.zero_grad()
    model_copy.train()

    # get copied named parameters
    copy_named  = [(n, p) for n, p in model_copy.named_parameters() 
                   if p.requires_grad]
    copy_names  = [n for n, _ in copy_named]
    params      = [p for _, p in copy_named]

    # check that nothing is out of order
    assert orig_names == copy_names, (
        "Parameter order mismatch!\n"
        f"orig: {orig_names}\n"
        f"copy: {copy_names}"
    )

    # initialize criterions
    ce_criterion = nn.CrossEntropyLoss() # L_A
    # L_K
    if args.unif_loss == "kl_forward" or args.unif_loss == "kl_reverse": 
        unif_criterion = nn.KLDivLoss(reduction='batchmean')
    if args.unif_loss == "square": 
        unif_criterion = torch.nn.MSELoss(reduction='mean')
    
    # L_K grad. Over D_f
    forget_processed = 0
    for forget_X, forget_y in forget_set_loader:
        forget_X, forget_y = forget_X.to(device), forget_y.to(device)
        
        outputs = model_copy(forget_X)
        softmax_outputs = F.softmax(outputs, dim=1)
        log_softmax_outputs = F.log_softmax(outputs, dim=1)
        
        num_classes = outputs.size(1)
        log_uniform = torch.full_like(outputs, -torch.log(torch.tensor(num_classes, dtype=outputs.dtype)))
        uniform = torch.full_like(softmax_outputs, 1.0/num_classes)
        
        if args.unif_loss == "kl_forward":
            unif_loss = unif_criterion(log_uniform, softmax_outputs) / len(forget_set_loader) # computes KL(softmax_outputs, unif)
        if args.unif_loss == "kl_reverse":
            unif_loss = unif_criterion(log_softmax_outputs, uniform) / len(forget_set_loader) # computes KL(unif, outputs)
        if args.unif_loss == "square":
            unif_loss = unif_criterion(softmax_outputs, uniform) * 0.5 * args.num_classes # rescaling 
        
        scaled_unif_loss = theta * unif_loss
        scaled_unif_loss.backward()
        forget_processed += 1

        del outputs, softmax_outputs, log_softmax_outputs, log_uniform, uniform, unif_loss, scaled_unif_loss
        torch.cuda.empty_cache()
    
    # L_A grad. over D_f as well
    retain_processed = 0
    for forget_X, forget_Y in forget_set_loader:
        forget_X, forget_Y = forget_X.to(device), forget_Y.to(device)
        
        outputs = model_copy(forget_X)
        ce_loss = ce_criterion(outputs, forget_Y) / len(forget_set_loader)
        scaled_ce_loss = (theta - 1) * (len(forget_set) / len(retain_set)) * ce_loss
        scaled_ce_loss.backward()
        retain_processed += 1
                            
        del outputs, ce_loss, scaled_ce_loss
        torch.cuda.empty_cache()
    
    # reg grad
    if args.with_reg: 
        for param in model_copy.parameters():
            if param.grad is not None:
                param.grad.add_(((len(forget_set) +  len(retain_set)) / len(retain_set)) *  lambda_reg * param.data)

            
    # return gradient
    grads = []
    for name, p in copy_named:
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grads.append(g.clone())
    return grads


# # Computes necessary HVP 
# # Credit to Binchi Zhang at https://github.com/zhangbinchi/certified-deep-unlearning/blob/main/unlearn.py
# # v is direction vector
# def hvp(loss, weights, v):
#     if len(weights) != len(v):
#         raise(ValueError("w and v must have the same length."))

#     # Compute first backprop
#     first_grads = grad(loss, weights, retain_graph=True, create_graph=True)

#     # Compute elementwise products
#     elemwise_products = 0
#     for grad_elem, v in zip(first_grads, v):
#         elemwise_products += torch.sum(grad_elem * v)

#     # Compute second backprop
#     return_grads = grad(elemwise_products, weights, create_graph=True) 

#     # Clean up
#     del first_grads, elemwise_products
#     torch.cuda.empty_cache()
#     gc.collect()

#     return return_grads

# Function to split data into retain set and forget set
def data_splitter(train_loader, train_eval_loader, args, num_workers=0, generator = None, worker_init_fn = None):
    
    # Extract the dataset from the loader
    dataset = train_loader.dataset
    eval_dataset = train_eval_loader.dataset
    total_size = len(dataset)
    forget_size = args.num_unif

    # Sample without replacement
    indices = torch.randperm(total_size, generator = generator)
    forget_indices = indices[:forget_size].tolist()
    retain_indices = indices[forget_size:].tolist()

    # Create subsets
    retain_train_subset = Subset(dataset, retain_indices)
    forget_train_subset = Subset(dataset, forget_indices)

    retain_eval_subset = Subset(eval_dataset, retain_indices)
    forget_eval_subset = Subset(eval_dataset, forget_indices)

    # Create DataLoaders
    retain_train_loader = DataLoader(retain_train_subset, 
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        num_workers=num_workers,
                        generator=generator,
                        worker_init_fn=worker_init_fn)

    forget_train_loader = DataLoader(forget_train_subset, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=num_workers,
                                        generator=generator,
                                        worker_init_fn=worker_init_fn)

    retain_eval_loader = DataLoader(retain_eval_subset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers, 
                                    generator=generator,
                                        worker_init_fn=worker_init_fn)

    forget_eval_loader = DataLoader(forget_eval_subset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers, 
                                    generator=generator,
                                    worker_init_fn=worker_init_fn)

    return retain_train_loader, retain_eval_loader, forget_train_loader, forget_eval_loader


def change_model_params(model, new_params):
    """
    Update model parameters with new parameters
    """
    updated_count = 0
    total_params = 0
    
    # Add debug output to check the keys and values
    print(f"Keys in new_params: {list(new_params.keys())}")
    
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            total_params += 1
            if param_name in new_params:
                if param.shape == new_params[param_name].shape:
                    # store original value for debugging
                    original_value = param.clone()
                    # update the param
                    param.copy_(new_params[param_name])
                    # verify param was changed
                    if torch.allclose(param, original_value):
                        print(f"Parameter {param_name} was not changed")
                    else:
                        updated_count += 1
                        #print(f"Updated {param_name}: different from original = {not torch.allclose(param, original_value)}")
                else:
                    print(f"Shape mismatch for {param_name}: {param.shape} vs {new_params[param_name].shape}")
            else:
                print(f"Parameter not found: {param_name}")
    
    print(f"Updated {updated_count}/{total_params} parameters")
    
    if updated_count == 0:
        print("WARNING: No parameters were updated!")
    
    return model

# Function to compute bound in Thm. 4.7; convex and nonconve modes
def compute_bound(forget_set, retain_set, args, theta, mode = "nonconvex", initial_grad = 0):
    if mode == "convex": 
        M_K = args.unif_hessian_lipschitz_M_K
        M_A = args.retain_hessian_lipschitz_M_A
        C = args.C
        M = theta * M_K + (1-theta) * M_A
        lambda_cvx = args.convex_approx_lambda

        bound = (2 * (C ** 2) * M) / lambda_cvx
        return bound
    elif mode == "nonconvex": 
        # forget set to get sample space dimension d and size of forget set, retain set size of retain set
        # initial_grad used to compute G
    
        C = args.C
        L_K = args.unif_grad_lipschitz_L_K
        L_A = args.retain_grad_lipschitz_L_A
        M_K = args.unif_hessian_lipschitz_M_K
        M_A = args.retain_hessian_lipschitz_M_A
        lambda_cvx = args.convex_approx_lambda
        lambda_min = args.min_eig_lambda_min
        zeta_min = args.min_eig_bound_indiv_zeta_min
        # flatten initial_grad into a vector
        flat_grad = torch.cat([g.contiguous().view(-1) for g in initial_grad])
        G = torch.norm(flat_grad, p=2) 
        rho = args.bound_looseness_rho
        d = np.prod(forget_set[0][0].shape[1:]) # ignoring batch dimension
        b = args.concentration_number_b
        n = args.sample_number_n

        B = np.max([((theta * L_K + lambda_cvx) / len(forget_set)),(((1-theta) * L_A + lambda_cvx) / len(retain_set)) ])
        
        supposed_n_size = 2 * (B / (lambda_cvx + lambda_min)) * np.log((B / (lambda_cvx + lambda_min)) * b)
        if (n < supposed_n_size):
            print(f"n: {n} is less than supposed n size: {supposed_n_size}")

        L = theta * L_K + (1-theta) * L_A
        M = theta * M_K + (1-theta) * M_A

        first_term = (2 * C) * (M * C + lambda_cvx) + G
        second_term = lambda_cvx + lambda_min
        third_term = 16 * (B / zeta_min) * np.sqrt(np.log(d / rho) / b) + 1/16
        fourth_term = 2 * C * L + G

        return ((first_term / second_term) + (third_term * fourth_term))
    else: 
        raise("Mode not supported")
    

# Function to compute standard deviation (to be used in Gaussian mechanism)
def compute_std(bound, args):
    return (bound / args.privacy_budget_epsilon) * np.sqrt(2 * np.log(1.25 / args.privacy_budget_delta))