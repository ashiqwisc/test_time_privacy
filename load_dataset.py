import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import random
import os

# try:
#     import medmnist
#     from medmnist import INFO
#     MEDMNIST_AVAILABLE = True
# except ImportError:
#     MEDMNIST_AVAILABLE = False
#     print("Warning: medmnist not installed. Install with: pip install medmnist")


def load_data(args, device, num_workers=0, generator=None, worker_init_fn=None):
    dataset_name = args.dataset
    # Convert images to tensors (automatically scales [0, 255] to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    cifar_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # Data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),  # Mean and std for CIFAR-10
    ])

    cifar_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train_eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])

    mean = []
    std  = []

    cifar100_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # Data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                            (0.2675, 0.2565, 0.2761)),  # Mean and std for CIFAR-100
    ])

    cifar100_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                            (0.2675, 0.2565, 0.2761)),  # Mean and std for CIFAR-100
    ])

    cifar100_train_eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), 
                            (0.2675, 0.2565, 0.2761)),  # Mean and std for CIFAR-100
    ])

    svhn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN stats
    ])
    
    if args.model_type == "ResNet50":
        print("resizing tinyimagenet to 224")
        tinyimagenet_train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),  
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                                (0.2302, 0.2265, 0.2262)),  
        ])

        tinyimagenet_test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                                (0.2302, 0.2265, 0.2262)),
        ])
    else:
        tinyimagenet_train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                                (0.2302, 0.2265, 0.2262)),  
        ])

        tinyimagenet_test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                                (0.2302, 0.2265, 0.2262)),
        ])
    imagenet_train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])

    imagenet_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])

    # Load the corresponding dataset
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_eval_dataset = train_dataset
    elif dataset_name == 'FMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        train_eval_dataset = train_dataset
    elif dataset_name == 'KMNIST':
        train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        train_eval_dataset = train_dataset
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_test_transform)
        train_eval_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,         # already downloaded
            transform=cifar_train_eval_transform
        )
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar100_train_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar100_test_transform)
        train_eval_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=False,         # already downloaded
            transform=cifar100_train_eval_transform
        )
    elif dataset_name == 'STL10': # make sure to load only labelled
        train_dataset = datasets.STL10(root='./data', split="train", download=True, transform=transform)
        test_dataset = datasets.STL10(root='./data', split="test", download=True, transform=transform)
    elif dataset_name == 'SVHN_standard':
        train_dataset = datasets.SVHN(root='./data', split="train", download=True, transform=svhn_transform)
        test_dataset = datasets.SVHN(root='./data', split="test", download=True, transform=svhn_transform)
        train_dataset.labels[train_dataset.labels == 10] = 0 # map labels 1-10 to 0-9 appropriately
        train_eval_dataset = datasets.SVHN(
            root='./data',
            split="train",
            download=False,         # already downloaded
            transform=svhn_transform
        )
        train_eval_dataset.labels[train_eval_dataset.labels == 10] = 0 
    elif dataset_name == 'SVHN_full':
        train_dataset = datasets.SVHN(root='./data', split="train", download=True, transform=svhn_transform)
        extra_dataset = datasets.SVHN(root='./data', split='extra', download=True, transform=svhn_transform)
        test_dataset = datasets.SVHN(root='./data', split="test", download=True, transform=svhn_transform)

        train_dataset.labels[train_dataset.labels == 10] = 0 # map labels 1-10 to 0-9 appropriately
        extra_dataset.labels[extra_dataset.labels == 10] = 0 # map labels 1-10 to 0-9 appropriately

        train_dataset = ConcatDataset([train_dataset, extra_dataset])

        train_eval_dataset = datasets.SVHN(
            root='./data',
            split="train",
            download=False,         # already downloaded
            transform=svhn_transform
        )

        extra_eval_dataset = datasets.SVHN(
            root='./data',
            split="extra",
            download=False,         # already downloaded
            transform=svhn_transform
        )

        train_eval_dataset.labels[train_eval_dataset.labels == 10] = 0 # map labels 1-10 to 0-9 appropriately
        extra_eval_dataset.labels[extra_eval_dataset.labels == 10] = 0 # map labels 1-10 to 0-9 appropriately

        train_eval_dataset = ConcatDataset([train_eval_dataset, extra_eval_dataset])
    
    
    # TinyImageNet 
    elif dataset_name == 'TinyImageNet':
        data_dir = os.path.abspath('data/tiny-imagenet-200')
        if not os.path.exists(data_dir):
            import urllib.request
            import zipfile
            print("Downloading TinyImageNet dataset...")
            os.makedirs('./data', exist_ok=True)
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            zip_path = './data/tiny-imagenet-200.zip'
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('./data')
            os.remove(zip_path)
            print("TinyImageNet downloaded successfully!")
        
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=tinyimagenet_train_transform
        )
        
        val_dir = os.path.join(data_dir, 'val')
        val_images_dir = os.path.join(val_dir, 'images')
        val_restructured = os.path.join(val_dir, 'restructured')
        
        if not os.path.exists(val_restructured):
            os.makedirs(val_restructured, exist_ok=True)
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_name = parts[0], parts[1]
                    class_dir = os.path.join(val_restructured, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    src = os.path.join(val_images_dir, img_name)
                    dst = os.path.join(class_dir, img_name)
                    if not os.path.exists(dst):
                        os.symlink(src, dst) if os.name != 'nt' else __import__('shutil').copy2(src, dst)
        
        test_dataset = datasets.ImageFolder(
            root=val_restructured,
            transform=tinyimagenet_test_transform
        )
        train_eval_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=tinyimagenet_test_transform
        )
    elif dataset_name == 'ImageNet1K':
        imagenet_path = "/home/xxiong52/dataset/imagenet1k"
        train_path = os.path.join(imagenet_path, "train")
        val_path = os.path.join(imagenet_path, "val")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"ImageNet train path not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"ImageNet val path not found: {val_path}")
        
        train_dataset = datasets.ImageFolder(train_path, transform=imagenet_train_transform)
        test_dataset = datasets.ImageFolder(val_path, transform=imagenet_test_transform)
        train_eval_dataset = datasets.ImageFolder(train_path, transform=imagenet_test_transform)
        
        print(f"ImageNet-1K loaded:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(test_dataset)}")
        print(f"  Number of classes: {len(train_dataset.classes)}")

    # elif dataset_name.lower() in [name.lower() for name in INFO.keys()] and MEDMNIST_AVAILABLE:

    #     medmnist_name = None
    #     for name in INFO.keys():
    #         if name.lower() == dataset_name.lower():
    #             medmnist_name = name
    #             break
        
    #     if medmnist_name:
    #         info = INFO[medmnist_name]
    #         n_channels = info['n_channels']
    #         DataClass = getattr(medmnist, info['python_class'])

    #         if n_channels == 1:  # Grayscale
    #             med_transform = transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[.5], std=[.5])
    #             ])
    #         else:  # RGB
    #             med_transform = transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    #             ])
            
    #         train_dataset = DataClass(
    #             split='train', 
    #             transform=med_transform, 
    #             download=True,
    #             root='./data'
    #         )
    #         test_dataset = DataClass(
    #             split='test', 
    #             transform=med_transform, 
    #             download=True,
    #             root='./data'
    #         )
    #         train_eval_dataset = DataClass(
    #             split='train', 
    #             transform=med_transform, 
    #             download=False,
    #             root='./data'
    #         )
    #     else:
            #raise ValueError(f"MedMNIST dataset '{dataset_name}' not found")
    
    else:
        # if dataset_name.lower() in [name.lower() for name in INFO.keys()] and not MEDMNIST_AVAILABLE:
        #     raise ImportError("MedMNIST not installed. Install with: pip install medmnist")
        # else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    if args.randomize_labels: # note: doesn't work for SVHN_full
        # Get the original labels, preferring 'targets' then 'labels'.
        original_labels = getattr(train_dataset, 'targets', getattr(train_dataset, 'labels', None))
        if original_labels is None:
            raise AttributeError("Dataset does not have 'targets' or 'labels' attribute.")

        # Create a random permutation of the labels.
        labels_tensor = torch.tensor(original_labels)
        shuffled_labels = labels_tensor[torch.randperm(len(labels_tensor))]

        # Assign the shuffled labels back to both datasets, preserving the original data type.
        if hasattr(train_dataset, 'targets'):
            new_labels = shuffled_labels.tolist()
            train_dataset.targets = new_labels
            train_eval_dataset.targets = new_labels
        elif hasattr(train_dataset, 'labels'):
            if isinstance(original_labels, list):
                new_labels = shuffled_labels.tolist()
            else: 
                new_labels = shuffled_labels.numpy() # assume numpy array
                train_dataset.labels = new_labels
                train_eval_dataset.labels = new_labels

    if args.downsample != 0: 
        num_total = len(train_dataset)
        print(num_total)
        num_subsample = num_total // args.downsample 
        subsample_indices = torch.randperm(num_total)[:num_subsample]

        train_dataset = torch.utils.data.Subset(train_dataset, subsample_indices)
        train_eval_dataset = torch.utils.data.Subset(train_eval_dataset, subsample_indices)
        print(len(train_eval_dataset))

        args.num_unif = args.num_unif // args.downsample # also decrease # of forget instances proportionally
        args.unif_batch_size = args.num_unif // args.downsample 
        
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              generator=generator)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # order doesn't matter for accuracy
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    return train_loader, test_loader, train_eval_loader, train_dataset, test_dataset