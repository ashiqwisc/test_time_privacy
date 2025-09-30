import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def get_nearest_neighbors_dataloader(forget_eval_loader, test_loader, device):
    # get forget samples from the forget set loader
    forget_inputs = []
    for x, _ in tqdm(forget_eval_loader, desc="Collecting forget set"):
        forget_inputs.append(x.to(device))
    forget_inputs = torch.cat(forget_inputs, dim=0)  

    # collect all test samples 
    test_inputs = []
    test_labels = []
    for x, y in tqdm(test_loader, desc="Collecting test set"):
        test_inputs.append(x.to(device))
        test_labels.append(y.to(device))
    test_inputs = torch.cat(test_inputs, dim=0)     
    test_labels = torch.cat(test_labels, dim=0)    

    # flatten 
    forget_flat = forget_inputs.view(forget_inputs.size(0), -1)  
    test_flat = test_inputs.view(test_inputs.size(0), -1)        

    # find nearest neighbors of forget set in test set 
    dists = torch.cdist(forget_flat, test_flat, p=2)
    nearest_indices = dists.argmin(dim=1)  # returns same # of indices as there are instances in forget set 
    nearest_neighbors = test_inputs[nearest_indices]
    nearest_labels = test_labels[nearest_indices]

    # return new dataloader 
    neighbor_dataset = TensorDataset(nearest_neighbors.cpu(), nearest_labels.cpu())
    neighbor_loader = DataLoader(neighbor_dataset, 
                batch_size=forget_eval_loader.batch_size,  # this is just gonna be used for eval, so no need to pass generator or worker fcn. etc. 
                shuffle=False)

    return neighbor_loader
