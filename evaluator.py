import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats
import matplotlib.pyplot as plt
from visualization import save_softmax_visualizations


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return 100 * correct / total


def forget_quality(model, forget_loader, img_save_path, model_type=None, batch_size=128, num_classes=10, args = None, seed_worker = None, g = None):
    """
    Evaluation metrics:
    1. Difference between accuracy and uniform 
    2. Uniformity loss divergence between model outputs and uniform distribution
    3. Save plots of softmax outputs for visualization
    
    Returns:
        results: Dictionary containing forget quality metrics
    """
    device = next(model.parameters()).device
    model.eval()
    
    dataloader = forget_loader
     
    # random guess accuracy 
    uniform_acc = (1 / num_classes) * 100
    
    correct = 0
    total = 0
    all_softmax_outputs = []
    all_labels = []
    unif_losses = []
    conf_distances = [] # the distance between max(softmax, 1/|num of classes|)

    if args.unif_loss == "kl_forward" or args.unif_loss == "kl_reverse": 
        unif_criterion = nn.KLDivLoss(reduction='batchmean')
    if args.unif_loss == "square": 
        unif_criterion = torch.nn.MSELoss(reduction='mean')
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            
            outputs = model(inputs) # logits output 
            softmax_outputs = F.softmax(outputs, dim=1) # probs output 
            log_softmax_outputs = F.log_softmax(outputs, dim=1) # log probs output 
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
           
            
            # Loss from uniform distribution
            num_classes = outputs.size(1)

            log_uniform = torch.full_like(outputs, -torch.log(torch.tensor(num_classes, dtype=outputs.dtype))) # log probs uniform
            uniform = torch.full_like(softmax_outputs, 1.0/num_classes) # probs uniform 
            
            
            if args.unif_loss == "kl_forward":
                unif_loss = unif_criterion(log_uniform, softmax_outputs) # computes KL(softmax_outputs, unif)
            if args.unif_loss == "kl_reverse":
                unif_loss = unif_criterion(log_softmax_outputs, uniform) # computes KL(unif, outputs)
            if args.unif_loss == "square":
                unif_loss = unif_criterion(softmax_outputs, uniform) * 0.5 * args.num_classes # rescaling
            #print("unif_loss: ", unif_loss)
            
            
            average_distance = distance_from_uniform(softmax_outputs, num_classes, args)
            conf_distances.append(average_distance)
            unif_losses.append(unif_loss)
            all_softmax_outputs.append(softmax_outputs.cpu())
            all_labels.append(targets.cpu())
    
    accuracy = 100 * correct / total
    accuracy_diff = max(0, accuracy - uniform_acc)
    # avg_unif_loss = torch.cat(unif_losses).mean().item()
    avg_unif_loss = torch.tensor(unif_losses, device = device).mean().item()
    avg_confidence_distance = torch.tensor(conf_distances, device = device).mean().item()
    
    # concatenate all outputs and labels
    all_softmax_outputs = torch.cat(all_softmax_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_confidence_distance = torch.tensor(conf_distances, device = device).mean().item()
    
    print(f"Accuracy on forget set: {accuracy:.2f}%")
    print(f"Uniform random accuracy: {uniform_acc:.2f}%")
    print(f"Difference from uniform: {accuracy_diff:.2f}%")
    print(f"Average Uniform loss from uniform: {avg_unif_loss:.4f}")
    print(f"Average Distance from uniform: {avg_confidence_distance:.4f}")
    
    # save visualization 
    if img_save_path:
        save_softmax_visualizations(
            all_softmax_outputs, 
            all_labels, 
            num_classes, 
            img_save_path, 
            model_type
        )
    
    
    results = {
        "accuracy": accuracy,
        "uniform_accuracy": uniform_acc,
        "accuracy_diff": accuracy_diff,
        "avg_unif_loss": avg_unif_loss,
        "avg_confidence_distance": avg_confidence_distance
    }
    
    return results


def distance_from_uniform(softmax_list, num_classes, args):
    if args.confidence_distance == "paper": 
        max_prob = torch.max(softmax_list, dim=1)[0]
        uniform = 1.0/num_classes
        total_distance = torch.maximum(torch.zeros_like(max_prob), max_prob - uniform)
        average_distance = total_distance.mean()
    elif args.confidence_distance == "l2": 
        uniform_dist = torch.ones_like(softmax_list) / num_classes  
        diffs = softmax_list - uniform_dist                    
        per_sample_l2 = torch.norm(diffs, p=2, dim=1)          
        average_distance = per_sample_l2.mean() 
    else:
        raise("Confidence distance not supported!")
    
    return average_distance