import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import time
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import copy
from uniformity_helper import exact_grad, fast_grad, compute_bound, compute_std
import gc
from models import ModelInitializer
from collections import OrderedDict

# Function to compute inverse Hessian (H + \lambda I)^{-1} exactly, and then matmul with gradient. Returns the product of the inverse Hessian and gradient
def exact_inv_hessian(forget_set_loader, retain_set_loader, pretrained_model, args, device, theta):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    
    # get L_K loss 
    forget_processed = 0
    total_scaled_unif_loss = 0
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
        total_scaled_unif_loss += scaled_unif_loss
        scaled_unif_loss.backward(create_graph = True)
        forget_processed += 1

        del outputs, softmax_outputs, log_softmax_outputs, log_uniform, uniform, unif_loss, scaled_unif_loss
        torch.cuda.empty_cache()
    
    # get L_A loss
    retain_processed = 0
    total_scaled_ce_loss = 0
    for retain_X, retain_y in retain_set_loader:
        retain_X, retain_y = retain_X.to(device), retain_y.to(device)
        
        outputs = model_copy(retain_X)
        ce_loss = ce_criterion(outputs, retain_y) / len(retain_set_loader)
        scaled_ce_loss = (1 - theta) * ce_loss
        total_scaled_ce_loss += scaled_ce_loss
        scaled_ce_loss.backward(create_graph = True)
        retain_processed += 1
                            
        del outputs, ce_loss, scaled_ce_loss
        torch.cuda.empty_cache()
    
    # add regularization if necessary
    if args.with_reg: 
        for param in model_copy.parameters():
            if param.grad is not None:
                param.grad.add_(lambda_reg * param.data)

    # get gradient
    print("Getting gradient")
    flat_grads = torch.cat([p.grad.view(-1) for p in params])  
    g = flat_grads.detach()  

    print("total scaled ce loss", total_scaled_ce_loss)
    print("total scaled unif loss", total_scaled_unif_loss)
    print("‖g‖ =", g.norm().item())


    # build hessian row by row
    n = flat_grads.numel()
    H = torch.zeros(n, n, device=device)
    for i in tqdm(range(n), desc = "Building Hessian"):
        # ∇_{θ} [ flat_grads[i] ] gives the i-th row of H
        row_i = torch.autograd.grad(flat_grads[i], params, retain_graph=True)
        H[i]  = torch.cat([r.view(-1) for r in row_i])

    # Form the inverse and multiply it by the gradient

    # To account for numerical error, symmetrize and shift min eigenval before adding lambda I 
    H = (H + H.T) / 2 
    eigvals = torch.linalg.eigvalsh(H)
    min_eig = eigvals.min().item()

    if min_eig < 0:
        H -= min_eig * torch.eye(H.size(0), device=H.device)
    
    I = torch.eye(n, device=device)
    lambda_I = lambda_reg * I
    H_reg = H + lambda_I

    # Check min eigenvalue before we run it through Cholesky 
    eigvals = torch.linalg.eigvalsh(H_reg)
    print("H_reg spectrum:  min=", eigvals.min().item(),
        " max=", eigvals.max().item(),
        " cond=", eigvals.max().item()/eigvals.min().item())

    #  H_inv = torch.inverse(H_reg)   # exact, O(n^3)
    # Compute inv w/ Cholesky (note that H_reg is p.d.)
    L = torch.linalg.cholesky(H_reg)
    n = H_reg.size(0)
    H_inv = torch.empty_like(H_reg)
    for i in tqdm(range(n), desc = "Inverting Hessian"):
        e_i = I[:, i].unsqueeze(1)  # basis vector
        # solves H_reg x = e_i  via Cholesky factorizaiton. Stacking these column wise gives inverse
        H_inv[:, i] = torch.cholesky_solve(e_i, L).squeeze(1)

    x = H_inv @ g   

    print("‖H⁻¹g‖ =", x.norm().item())

    # Unflatten back into a list, same order as params
    inv_hess_grad_prod = []
    offset = 0
    for _, param in copy_named:     # copy_named is [(name, param), …]
        cnt = param.numel()
        inv_hess_grad_prod.append(
            x[offset:offset+cnt]
             .view_as(param)
             .clone()
        )
        offset += cnt

    inv_hess_updates = OrderedDict(zip(copy_names, inv_hess_grad_prod))

    return inv_hess_updates


# Function to induce uniformity on the forget set, utility on retain set, using exact Hessian computation
def exact_certified_uniformity(pretrained_model, 
                        retain_loader,
                        forget_loader,
                        args,
                        device, theta):
    
    start = time.time()

    inv_hess_grad_prod = exact_inv_hessian(forget_loader, retain_loader, pretrained_model, args, device, theta)


    bound = compute_bound(forget_loader.dataset, retain_loader.dataset, args, theta, mode = "convex", initial_grad = 0)
    print(f"Size of bound: {bound}")
    std = 1e-3
    print(f"Size of std: {std}")

    # initialize new model
    initalizer = ModelInitializer(device)
    certified_model = initalizer.init_model(args)
    certified_model.load_state_dict(pretrained_model.state_dict())

    with torch.no_grad():
        for name, param in certified_model.named_parameters():
            update = 0.5 * inv_hess_grad_prod[name] 
            # update = inv_hess_grad_prod[name] 
            noise  = std * torch.randn_like(param)
            param.add_(-update + noise)

    # compute the total difference between the original parameters and the updated parameters
    total_diff = 0.0
    total_count = 0   
    
    for (name_1, original_param), (name_2, updated_param) in zip(pretrained_model.named_parameters(), certified_model.named_parameters()):
        diff = torch.abs(updated_param - original_param)
        total_diff += diff.sum().item()        
        total_count += diff.numel()        

    # Compute the overall mean difference.
    mean_difference = total_diff / total_count
    print(f"Mean difference in uniformity.py: {mean_difference}")

    print(f'Time: {time.time() - start}')

    return certified_model