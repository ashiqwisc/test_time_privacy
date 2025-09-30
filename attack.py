import numpy as np
import torch
import argparse
import random
from tqdm import tqdm
import yaml
import os
from torch.utils.data import TensorDataset, DataLoader
from load_dataset import load_data
from uniformity_helper import data_splitter
from models import ModelInitializer
import torch.nn as nn
import torch.nn.functional as F
from evaluator import distance_from_uniform, evaluate, forget_quality
import json
from datetime import datetime

# add simple gaussian noise to instance, sampled uniformly at random, of size epsilon
def gauss(model, x, epsilon, alpha = 0): # alpha irrelevant here
    # Generate Gaussian noise with same shape as x
    noise = torch.randn_like(x) * epsilon
    
    # Add noise and clamp to valid image range
    x_perturbed = x + noise
    x_perturbed = torch.clamp(x_perturbed, 0.0, 1.0)
    
    return x_perturbed.detach()
     
# FGSM-style perturbation to maximize confidence in epsilon ball
def fgsm(model, x, epsilon, alpha=1e-4):
    # break symmetry with small random perturbation
    alpha0 = alpha*epsilon
    delta0 = torch.empty_like(x).uniform_(-alpha0, alpha0)
    x0 = (x + delta0).clone().detach().requires_grad_(True)

    x0.retain_grad()
    x0_clamped = torch.clamp(x0, 0.0, 1.0) # ensure in valid range

    # forward pass at x0 to compute maximum logit via logsumexp
    z = model(x0_clamped)  # logits
    # z = F.softmax(z, dim = 1) # take softmax first! 
    lse = torch.logsumexp(z, dim=1)  
    
    # compute gradient in direction of max confidence
    lse.backward(torch.ones_like(lse))
    grad_rho = x0.grad.detach()  
    
    # take FGSM step 
    delta = epsilon * torch.sign(grad_rho)
    x_adv = x + delta  # perturb original x
    x_adv = torch.clamp(x_adv, 0.0, 1.0) # clamp to valid image range
    
    return x_adv.detach()

def pgd(model, x, epsilon, alpha=1e-4, beta=1e-3, steps=50): 
    # break symmetry with small random perturbation
    alpha0 = alpha*epsilon
    delta0 = torch.empty_like(x).uniform_(-alpha0, alpha0)
    x0 = (x + delta0).clone().detach().requires_grad_(True)
    x_adv = torch.clamp(x0, 0.0, 1.0)   # clamp to valid image range
    
    for _ in range(steps):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        
        # grad ascent step to max. confidence 
        z = model(x_adv)
        lse = torch.logsumexp(z, dim=1)  
        model.zero_grad()
        lse.backward(torch.ones_like(lse))
        grad = x_adv.grad.detach()

        with torch.no_grad():
            x_adv = x_adv + beta * torch.sign(grad)
            
            # project back to epsilon ball and valid image range 
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    return x_adv.detach()

# get new dataloader with attacked instances
def create_perturbed_dataloader(model, dataloader, epsilon, attack, num_workers=0, generator=None, worker_init_fn=None, alpha=1e-4):

    commands = {
        'fgsm': fgsm,
        'pgd': pgd,
        'gauss': gauss
    }

    attack = commands[attack]
    
    device = next(model.parameters()).device
    model.eval()
    
    perturbed_inputs = []
    original_labels = []
    
    for inputs, labels in tqdm(dataloader, 
                             desc="Generating perturbations", 
                             unit="batch",
                             leave=True):
        inputs = inputs.to(device)
        
        # generate perturbed batch
        with torch.enable_grad():  # Enable gradient computation for inputs
            perturbed_batch = attack(model, inputs, epsilon, alpha)
        
        # store results
        perturbed_inputs.append(perturbed_batch.cpu())
        original_labels.append(labels)
    
    # combine all batches
    all_perturbed = torch.cat(perturbed_inputs, dim=0)
    all_labels = torch.cat(original_labels, dim=0)
    
    # create new dataset and dataloader
    perturbed_dataset = TensorDataset(all_perturbed, all_labels)
    perturbed_loader = DataLoader(
        perturbed_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # w/ original order
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        generator = generator, 
        worker_init_fn=worker_init_fn
    )
    
    return perturbed_loader
