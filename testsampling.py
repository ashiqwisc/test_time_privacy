from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import random
import torchvision.transforms.functional as F
from torchvision import datasets, transforms

class TransformOverrideDataset(Dataset):
    def __init__(self, base_dataset, new_transform):
        self.base = base_dataset
        self.override_tf = new_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = F.to_pil_image(x)
        x = self.override_tf(x) # apply the override instead of base.transform
        return x, y

def test_sampling(frac_of_test, train_loader, train_eval_loader, forget_train_loader, forget_eval_loader, test_loader, args,
                    num_workers=0, generator=None, worker_init_fn=None): 
    

    # test and eval datasets already transformed, no need to port over from load_dataset.py
    if args.dataset == "CIFAR10": 
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # Data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                (0.2023, 0.1994, 0.2010)),  # Mean and std for CIFAR-10
        ])
    elif args.dataset == "CIFAR100": 
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # Data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                (0.2675, 0.2565, 0.2761)),  # Mean and std for CIFAR-100
        ])
    elif args.dataset == "SVHN": 
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN stats
        ])
    elif args.dataset == "TinyImageNet": 
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                                (0.2302, 0.2265, 0.2262)),  
        ])
    else: 
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    num_forget = len(forget_train_loader.dataset)
    num_to_sample = int(num_forget * frac_of_test)
    test_dataset = test_loader.dataset # doesn't have train set transform rn 
    test_size = len(test_dataset)

    print(num_forget)
    print(test_size)
    print(num_to_sample)
    sample_indices = random.sample(range(test_size), num_to_sample)  # sample test set instances w/out replacement; deterministic by seed setting 

    train_subset = Subset(train_loader.dataset, list(range(len(train_loader.dataset))))
    train_eval_subset = Subset(train_eval_loader.dataset, list(range(len(train_eval_loader.dataset))))
    forget_train_subset = Subset(forget_train_loader.dataset, list(range(num_forget)))
    forget_eval_subset = Subset(forget_eval_loader.dataset, list(range(num_forget)))

    test_eval_subset  = Subset(test_loader.dataset, sample_indices) # ensured new forget set (used for finetuning) inherits transform
    test_train_subset = TransformOverrideDataset(test_eval_subset, train_transform)

    joined_train_set =  ConcatDataset([train_subset, test_train_subset])
    joined_train_eval_set =  ConcatDataset([train_eval_subset, test_eval_subset])
    joined_forget_train_set = ConcatDataset([forget_train_subset, test_train_subset])
    joined_forget_eval_set = ConcatDataset([forget_eval_subset, test_eval_subset])

    new_train_loader =  DataLoader( # we will finetune pretrained model w/ these instances before running Alg. 1 
        joined_train_set, 
        # test_train_subset,
        batch_size=args.batch_size,
        shuffle=True,    
        num_workers=num_workers, 
        generator=generator, 
        worker_init_fn=worker_init_fn
    )

    new_train_eval_loader =  DataLoader( # during finetuning, we will eval on this 
        joined_train_eval_set,
        # test_eval_subset,
        batch_size=args.batch_size,
        shuffle=False,    
        num_workers=num_workers, 
        generator=generator, 
        worker_init_fn=worker_init_fn
    )

    new_forget_train_loader = DataLoader( # will then run Alg. 1 on this + existing retain loader
        joined_forget_train_set,
        batch_size=args.unif_batch_size,
        shuffle=True,    
        num_workers=num_workers, 
        generator=generator, 
        worker_init_fn=worker_init_fn
    )

    new_forget_eval_loader = DataLoader( # during Alg. 1, we will eval on this + existing retain loader
        joined_forget_eval_set,
        batch_size=args.unif_batch_size,
        shuffle=False,    
        num_workers=num_workers, 
        generator=generator, 
        worker_init_fn=worker_init_fn
    )

    # remove sampled test data from this one, make sure it doesn't inherit transform
    all_test_idxs = set(range(len(test_loader.dataset)))
    sampled_test_idxs = set(sample_indices)
    remaining_test_idxs = list(all_test_idxs - sampled_test_idxs)
    test_remaining_subset = Subset(test_loader.dataset, remaining_test_idxs)
    new_test_loader = DataLoader(
        test_remaining_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn
    )

    return new_train_loader, new_train_eval_loader, new_forget_train_loader, new_forget_eval_loader, new_test_loader