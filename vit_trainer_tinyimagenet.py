import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from timm import create_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchvision.transforms import RandAugment
from timm.layers import DropPath
import copy

class ViTTrainer:
    def __init__(self, device):
        self.device = device
        
    def create_vit_model(self, num_classes=200):
        model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self._apply_stochastic_depth(model, drop_prob=0.1)  
        # unfreeze the entire model for fine-tuning
        for param in model.parameters():
            param.requires_grad = True           
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
    
    def get_vit_transforms(self, image_size=224):
        """Get transforms optimized for ViT on TinyImageNet"""
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                               (0.2302, 0.2265, 0.2262)),
            transforms.RandomErasing(p=0.25),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), 
                               (0.2302, 0.2265, 0.2262)),
        ])
        
        return transform_train, transform_test
    
    def retrain_vit_from_scratch(self, retain_loader, retain_val_loader, args, test_loader=None):
        print("Training ViT model from scratch on retain set...")

        model = self.create_vit_model(num_classes=args.num_classes)
   
        learning_rate = 1e-4
        weight_decay = 0.01
        num_epochs = args.epochs if hasattr(args, 'epochs') else 50
        
        # Mixup and CutMix for better training
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=0.5,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=0.1,
            num_classes=args.num_classes
        )
        
        # Loss and optimizer
        criterion = SoftTargetCrossEntropy()  # For Mixup and CutMix
        criterion_val = nn.CrossEntropyLoss()  # Standard loss for validation
        
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = torch.amp.GradScaler(device='cuda')
        
        best_acc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            train_loader_tqdm = tqdm(retain_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
            
            for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # Apply Mixup/CutMix
                images, labels = mixup_fn(images, labels)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                
                # Calculate accuracy (for soft labels, use argmax)
                _, predicted = outputs.max(1)
                _, targets = labels.max(1)
                correct += predicted.eq(targets).sum().item()
                
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
                        
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion_val(outputs, labels)
                        
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
  
        if best_model_state is not None:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(best_model_state)
            else:
                model.load_state_dict(best_model_state)
        
        print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
        return model