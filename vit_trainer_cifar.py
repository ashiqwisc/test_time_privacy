import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from timm.layers import DropPath
import copy

class CIFAR100ViTTrainer:
    def __init__(self, device):
        self.device = device
        
    def create_cifar100_vit_model(self, model_type="ViT-S_16", num_classes=100):
        if model_type == "ViT-S_16":
            model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)
            print("Loaded timm pretrained ViT-S/16 for CIFAR-100")
        elif model_type == "ViT-B_16":
            model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
            print("Loaded timm pretrained ViT-B/16 for CIFAR-100")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Apply Stochastic Depth (same as TinyImageNet)
        self._apply_stochastic_depth(model, drop_prob=0.1)
        
        # Use DataParallel for multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
            
        return model.to(self.device)
    
    def _apply_stochastic_depth(self, model, drop_prob):
        """Apply stochastic depth to improve regularization during training"""
        for module in model.modules():
            if isinstance(module, DropPath):
                module.drop_prob = drop_prob
    
    def get_cifar100_transforms(self):
        """Get transforms for CIFAR-100 (resize to 224 for ViT)"""
        transform_train = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 stats
            transforms.RandomErasing(p=0.25),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        return transform_train, transform_test
    
    def retrain_cifar100_vit_from_scratch(self, retain_loader, retain_val_loader, args, test_loader=None):
        """
        Retrain ViT from scratch on CIFAR-100 using the retain set only
        Using timm models with similar configuration as TinyImageNet trainer
        """
        print(f"Training {args.model_type} model from scratch on CIFAR-100 retain set...")
        
        # Create model
        model = self.create_cifar100_vit_model(model_type=args.model_type, num_classes=args.num_classes)
        
        # Setup training parameters (similar to TinyImageNet approach)
        learning_rate = 1e-4  # Conservative learning rate for fine-tuning
        weight_decay = 0.01
        num_epochs = args.epochs if hasattr(args, 'epochs') else 50
        
        # Loss function and optimizer (same as TinyImageNet)
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = GradScaler()
        
        best_acc = 0
        best_model_state = None
        
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            train_loader_tqdm = tqdm(retain_loader, 
                                   desc=f"Epoch {epoch+1}/{num_epochs} [Training]", 
                                   leave=False)
            
            for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                if (batch_idx + 1) % 100 == 0:
                    current_loss = loss.item()
                    current_acc = 100. * correct / total
                    train_loader_tqdm.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.2f}%")
            
            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
            # Validation phase
            if retain_val_loader is not None:
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                
                with torch.no_grad():
                    for images, labels in retain_val_loader:
                        images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                        
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * images.size(0)
                        val_total += labels.size(0)
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(labels).sum().item()
                
                val_acc = 100. * val_correct / val_total
                print(f"Validation Loss: {val_loss/val_total:.4f}, Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    if isinstance(model, nn.DataParallel):
                        best_model_state = copy.deepcopy(model.module.state_dict())
                    else:
                        best_model_state = copy.deepcopy(model.state_dict())
                    print(f"New best model with accuracy: {best_acc:.2f}%")
            
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(best_model_state)
            else:
                model.load_state_dict(best_model_state)
        
        print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
        return model