import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
from uniformity_helper import data_splitter
import random
from load_dataset import load_data
from train import ModelTrainer
from models import ModelInitializer
import copy
from tqdm import tqdm
import os

def subsample_from_loader(loader, total, rng):
    # extract full dataset and targets from loader
    dataset = loader.dataset

    # handle Subset or Dataset directly
    if isinstance(dataset, Subset):
        all_indices = np.array(dataset.indices)
        targets = np.array(dataset.dataset.targets)[all_indices]
    else:
        all_indices = np.arange(len(dataset))
        targets = np.array(dataset.targets)

    # filter indices by class
    idx0 = all_indices[targets == 0]
    idx1 = all_indices[targets == 1]
    n_each = total // 2
    assert len(idx0) >= n_each and len(idx1) >= n_each, \
        f"Not enough samples: need {n_each} of each class"

    chosen0 = rng.choice(idx0, size=n_each, replace=False)
    chosen1 = rng.choice(idx1, size=n_each, replace=False)
    chosen = np.concatenate([chosen0, chosen1])
    rng.shuffle(chosen)

    return Subset(dataset if not isinstance(dataset, Subset) else dataset.dataset, chosen.tolist())
    
def downsample(train_loader, test_loader, train_eval_loader, args, device, num_workers=0, generator=None, worker_init_fn=None):
    rng = np.random.RandomState(args.seed)

    # create subsampled Subsets
    sub_train_ds = subsample_from_loader(train_loader, 1000, rng)
    sub_test_ds  = subsample_from_loader(test_loader,  500, rng)
    sub_train_eval_ds = subsample_from_loader(train_eval_loader, 1000, rng)

    # build DataLoaders
    sub_train_loader = DataLoader(sub_train_ds, batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers,
                                  generator=generator,
                                  worker_init_fn=worker_init_fn)
    sub_train_eval_loader = DataLoader(sub_train_ds, batch_size=args.batch_size, 
                                  shuffle=False,
                                  num_workers=num_workers,
                                  generator=generator,
                                  worker_init_fn=worker_init_fn)

    sub_test_loader = DataLoader(sub_test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers,
                                 generator=generator,
                                 worker_init_fn=worker_init_fn)

    return sub_train_loader, sub_train_eval_loader, sub_test_loader


# gets grad, but not L_K + L_A grad like exact_grad; one at a time
def get_grad(pretrained_model, loader, args, type_loss, device):
    pretrained_model.zero_grad()
    if type_loss == "K": 
        criterion = nn.KLDivLoss(reduction='batchmean')
        for forget_X, forget_y in loader:
            forget_X, forget_y = forget_X.to(device), forget_y.to(device)
            
            outputs = pretrained_model(forget_X)
            softmax_outputs = F.softmax(outputs, dim=1)
            log_softmax_outputs = F.log_softmax(outputs, dim=1)
            
            num_classes = outputs.size(1)
            log_uniform = torch.full_like(outputs, -torch.log(torch.tensor(num_classes, dtype=outputs.dtype)))
            uniform = torch.full_like(softmax_outputs, 1.0/num_classes)
            
            unif_loss = criterion(log_uniform, softmax_outputs) / len(loader)
            unif_loss.backward(create_graph = True)

            del outputs, softmax_outputs, log_softmax_outputs, log_uniform, uniform, unif_loss
            torch.cuda.empty_cache()

        for param in pretrained_model.parameters():
            if param.grad is not None:
                param.grad.add_(args.convex_approx_lambda * param.data)
    else: 
        criterion = nn.CrossEntropyLoss() 
        for retain_X, retain_y in loader:
            retain_X, retain_y = retain_X.to(device), retain_y.to(device)
            
            outputs = pretrained_model(retain_X)

            ce_loss = criterion(outputs, retain_y) / len(loader)
            ce_loss.backward(create_graph = True)
                                
            del outputs, ce_loss
            torch.cuda.empty_cache()

        for param in pretrained_model.parameters():
            if param.grad is not None:
                param.grad.add_(args.convex_approx_lambda * param.data)

    flat_grads = torch.cat([p.grad.view(-1) for p in pretrained_model.parameters()])  
    g0 = flat_grads.detach() 

    return g0

# gets lipschitz constant of grad
def get_grad_L(pretrained_model, loader, args, type_loss, device, delta=1e-3, n_dirs=5):
    g0 = get_grad(pretrained_model, loader, args,type_loss, device)
    max_ratio = 0.0

    for _ in range(n_dirs):
        # get dir epsilon of norm delta
        eps = np.random.randn(*g0.shape)
        eps *= delta / np.linalg.norm(eps)

        # perturb parameters
        with torch.no_grad():
            orig_linear = model.linear.weight.data.clone()
            orig_bias = model.linear.bias.data.clone()

            weight_shape = model.linear.weight.shape  
            bias_shape = model.linear.bias.shape      
            weight_numel = model.linear.weight.numel()
            bias_numel = model.linear.bias.numel()

            eps_tensor = torch.from_numpy(eps).to(device)

            model.linear.weight.data += eps_tensor[:weight_numel].reshape(weight_shape)
            model.linear.bias.data += eps_tensor[weight_numel:].reshape(bias_shape)

        g1 = get_grad(pretrained_model, loader, args, type_loss, device)
        ratio = torch.norm(g1 - g0) / delta
        max_ratio = max(max_ratio, ratio)

        # reset
        with torch.no_grad():
            model.linear.weight.data.copy_(orig_linear)
            model.linear.bias.data.copy_(orig_bias)

    return max_ratio

# need the same function that takes in a loss, pretrained model, a dataloader, and returns the lipschitz constant of the hessian
def get_hess(pretrained_model, loader, args, type_loss, device):
    pretrained_model.zero_grad()
    if type_loss == "K": 
        criterion = nn.KLDivLoss(reduction='batchmean')
        for forget_X, forget_y in loader:
            forget_X, forget_y = forget_X.to(device), forget_y.to(device)
            
            outputs = pretrained_model(forget_X)
            softmax_outputs = F.softmax(outputs, dim=1)
            log_softmax_outputs = F.log_softmax(outputs, dim=1)
            
            num_classes = outputs.size(1)
            log_uniform = torch.full_like(outputs, -torch.log(torch.tensor(num_classes, dtype=outputs.dtype)))
            uniform = torch.full_like(softmax_outputs, 1.0/num_classes)
            
            unif_loss = criterion(log_uniform, softmax_outputs) / len(loader)
            unif_loss.backward(create_graph = True) # no need for theta here

            del outputs, softmax_outputs, log_softmax_outputs, log_uniform, uniform, unif_loss
            torch.cuda.empty_cache()

        for param in pretrained_model.parameters():
            if param.grad is not None:
                param.grad.add_(args.convex_approx_lambda * param.data)
    else: 
        criterion = nn.CrossEntropyLoss() 
        for retain_X, retain_y in loader:
            retain_X, retain_y = retain_X.to(device), retain_y.to(device)
            
            outputs = pretrained_model(retain_X)

            ce_loss = criterion(outputs, retain_y) / len(loader)
            ce_loss.backward(create_graph = True)
                                
            del outputs, ce_loss
            torch.cuda.empty_cache()

        for param in pretrained_model.parameters():
            if param.grad is not None:
                param.grad.add_(args.convex_approx_lambda * param.data)

    flat_grads = torch.cat([p.grad.view(-1) for p in pretrained_model.parameters()])     

    n = flat_grads.numel()
    H = torch.zeros(n, n, device=device)
    for i in tqdm(range(n), desc = "Building Hessian"):
        # gives ith row of H
        row_i = torch.autograd.grad(flat_grads[i], pretrained_model.parameters(), retain_graph=True)
        H[i]  = torch.cat([r.view(-1) for r in row_i])

    return H

def get_hess_L(pretrained_model, loader, args, type_loss, device, delta=1e-3, n_dirs=5):
    H0 = get_hess(pretrained_model, loader, args, type_loss, device)
    max_ratio = 0.0
    for _ in range(n_dirs):
        # get eps direction similarly
        D = H0.shape[0]
        eps = np.random.randn(D)
        eps *= delta / np.linalg.norm(eps)
        # perturb simularly 
        with torch.no_grad():
            orig_linear = model.linear.weight.data.clone()
            orig_bias = model.linear.bias.data.clone()

            weight_shape = model.linear.weight.shape  
            bias_shape = model.linear.bias.shape      
            weight_numel = model.linear.weight.numel()
            bias_numel = model.linear.bias.numel()

            eps_tensor = torch.from_numpy(eps).to(device)

            model.linear.weight.data += eps_tensor[:weight_numel].reshape(weight_shape)
            model.linear.bias.data += eps_tensor[weight_numel:].reshape(bias_shape)

        H1 = get_hess(pretrained_model, loader, args, type_loss, device)

        H_change = H1 - H0

        # spectral norm (max abs eigenvalue, since symmetric)
        eigs = np.linalg.eigvalsh(H_change.cpu())
        norm = np.max(np.abs(eigs))
        ratio = norm / delta
        max_ratio = max(max_ratio, ratio)

        # reset model
        with torch.no_grad():
            model.linear.weight.data.copy_(orig_linear)
            model.linear.bias.data.copy_(orig_bias)

    return max_ratio

def get_ce_loss(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')  # sum over batch
    total_ce_loss = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            total_ce_loss += criterion(outputs, y).item()  # sum of CE losses
            total_samples += y.size(0)  # number of samples in batch
            torch.cuda.empty_cache()

    return total_ce_loss / total_samples  # average CE loss

print("PID:", os.getpid())

# prepare args so we can train (ensure that we train w/ PGD and appropriate C and lambda)
args = argparse.Namespace()
args.batch_size = 128
args.epochs = 1 #relevant 
args.lr = 0.01 #relevant 
args.dataset = "MNIST"
args.model_type = "LogisticRegression"
args.train = True 
args.num_layer = 2 
args.hidden_dim = 100 
args.num_classes = 10 
args.seed = 42 # relevant 
args.trained_model_save_path = ""
args.trained_model_load_path = ""
args.unif_model_save_path = ""
args.forget_images_path = ""
args.results_file_path = ""
args.num_unif = 10
args.unif_batch_size = 10 
args.confidence_distance = "paper"
args.C = 10 #relevant 
args.with_pgd = True # relevant 
args.concentration_number_b = 0
args.sample_number_n = 0
# Note: We use a slightly less tight bound than our paper bound by bounding sqrt above--valid bc sqrt mtn increasing, removes condition on lambda
args.convex_approx_lambda = 0.05 # relevant  
args.min_eig_lambda_min = 0
args.min_eig_bound_indiv_zeta_min = 0
args.scale_Hessian_H = 0
args.privacy_budget_epsilon = 0
args.privacy_budget_delta = 0
args.compute_theta = False 
args.distance_from_unif = False
args.util_unif_tradeoff_theta = 1 # relevant 
args.use_schedule = False 
args.util_unif_init = 1
args.bound_looseness_rho = 0
args.unif_grad_lipschitz_L_K = 0 
args.retain_grad_lipschitz_L_A = 0 
args.unif_hessian_lipschitz_M_K = 0
args.retain_hessian_lipschitz_M_A = 0
args.run_retrain = False 
args.run_pareto = True # relevant 
args.uniformity_mode = "skip"
args.pareto_epochs = 100 # relevant
args.unif_loss = "kl_forward" # relevant 
args.with_reg = True # relevant 
args.warmup = False 
args.warmup_reg = 0.0 
args.surgery = True # relevant 

# set seeds etc.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device:{device}")

pareto_device = device

# set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)
num_workers = 2

# will load MNIST 
train_loader, test_loader, train_eval_loader, train_dataset, test_dataset = load_data(args, device, num_workers=num_workers, generator=g, worker_init_fn=seed_worker)

# downsample MNIST to 1000 instances of 0 or 1
# train_loader, train_eval_loader, test_loader  = downsample(train_loader, test_loader, train_eval_loader, args, device, num_workers=num_workers, generator=g, worker_init_fn=seed_worker)

# get retrain and forget sets
retain_train_loader, retain_eval_loader, forget_train_loader, forget_eval_loader = data_splitter(train_loader, train_eval_loader, args, num_workers=num_workers, generator=g, worker_init_fn=seed_worker)

# get pretrained model on retain set 
initalizer = ModelInitializer(device)
model = initalizer.init_model(args)
trainer = ModelTrainer(device)
data_shape = [3,28,28] # MNIST 
pretrained_model = trainer.retrain_from_scratch(model=model, 
                    retain_loader=retain_train_loader,   
                    retain_val_loader=retain_eval_loader,
                    args=args, 
                    num_workers=num_workers, 
                    seed_worker=seed_worker, 
                    g=g,
                    data_shape=data_shape,
                    logistic=False)

# get pareto_finetune model on both retain and forget sets
pareto_finetune_model, pareto_durs = trainer.train_pareto_model(model=copy.deepcopy(pretrained_model),
                                forget_loader = forget_train_loader,
                                retain_loader = retain_train_loader,
                                retain_val_loader = retain_eval_loader,
                                forget_val_loader = forget_eval_loader,
                                test_loader=test_loader,
                                device=device,
                                args=args,
                                seed_worker = seed_worker,
                                g = g,
                                reg = args.with_reg)

# compute alpha^* and alpha(args.util_unif_tradeoff_theta)
alpha_star = get_ce_loss(pretrained_model, retain_eval_loader, device=device)
alpha_theta = get_ce_loss(pareto_finetune_model,  retain_eval_loader, device=device) 

# estimate Lipschitz constants (at w^* = A(D))
L_K = get_grad_L(pretrained_model, forget_eval_loader, args, type_loss="K", device=device)
M_K = get_hess_L(pretrained_model, forget_eval_loader, args, type_loss="K", device=device)
L_A = get_grad_L(pretrained_model, forget_eval_loader, args, type_loss="A", device=device)
M_A = get_hess_L(pretrained_model, forget_eval_loader, args, type_loss="A", device=device)

# get params from args
lambda_ = args.convex_approx_lambda
C       = args.C 
L = args.util_unif_tradeoff_theta * L_K + (1-args.util_unif_tradeoff_theta) * L_A
M = args.util_unif_tradeoff_theta * M_K + (1-args.util_unif_tradeoff_theta) * M_A

# compute args.util_unif_tradeoff_theta = 1 bound 
delta   = lambda_ - L
disc    = delta**2 - 4*args.util_unif_tradeoff_theta*C*M*(2*L_K + lambda_)
term    = abs(delta - abs(disc)) # got rid of sqrt here
B1      = (L_A / (4*M)) * term
B2      = (lambda_*C / (2*M)) * term
bound   = B1 + B2

print(f"alpha*: {alpha_star:.4f}, alpha(args.util_unif_tradeoff_theta): {alpha_theta:.4f}, |a^* - alpha(args.util_unif_tradeoff_theta)|: {abs(alpha_star - alpha_theta):.4f}")
print(f"Estimated constants: L_K={L_K:.4f}, M_K={M_K:.4f}, L_A={L_A:.4f}, M_A={M_A:.4f} ")
print(f"Theoretical bound: {bound:.4f}")
