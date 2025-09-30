import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoImageProcessor, ResNetForImageClassification

class ModelInitializer:
    def __init__(self, device):
        self.device = device
    
    def init_model(self, args):
        print(args.model_type, args.dataset)
        if args.dataset == "MNIST" or args.dataset == "FMNIST" or args.dataset == "KMNIST":
            input_channels = 1
            img_size = 28
            input_dim = input_channels * img_size * img_size 
        elif args.dataset == "CIFAR100" or args.dataset == "CIFAR10" or args.dataset == "CIFAR5":
            input_channels = 3
            img_size = 32
            input_dim = input_channels * img_size * img_size  
        elif args.dataset == "STL10":
            input_channels = 3
            img_size = 96
            input_dim = input_channels * img_size * img_size  
        elif args.dataset == "SVHN_standard" or args.dataset == "SVHN_full":
            input_channels=3
            img_size = 32
            input_dim = input_channels * img_size * img_size
        elif args.dataset == "TinyImageNet":
            input_channels = 3
            img_size = 64
            input_dim = input_channels * img_size * img_size
        elif args.dataset == "ImageNet1K":
            input_channels = 3
            img_size = 224  # ImageNet standard size
            input_dim = input_channels * img_size * img_size
        #elif args.dataset.lower() in [name.lower() for name in INFO.keys()] and MEDMNIST_AVAILABLE:
            # # Handle all MedMNIST datasets
            # medmnist_name = None
            # for name in INFO.keys():
            #     if name.lower() == args.dataset.lower():
            #         medmnist_name = name
            #         break
            
            # if medmnist_name:
            #     info = INFO[medmnist_name]
            #     input_channels = info['n_channels']  # 1 for grayscale, 3 for RGB
            #     img_size = 28  # All MedMNIST datasets are 28x28
            #     input_dim = input_channels * img_size * img_size
            # else:
            #     raise ValueError(f"MedMNIST dataset '{args.dataset}' not found")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        if args.model_type == "MLP":
            print("==================================\nInitializing MLP\n=====================================")
            model = MLP(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_classes=args.num_classes,
                num_layer=args.num_layer
            )
        elif args.model_type == "LogisticRegression":
            print("==================================\nInitializing Logistic Regression\n==================================")
            model = LogisticRegression(
                input_dim = input_dim,
                num_classes = args.num_classes
            )
        elif args.model_type == "ResNet8":
            print("==================================\nInitializing ResNet4\n==================================")
            model = ResNet8(
                num_channels = input_channels,
                num_classes = args.num_classes
            )
        elif args.model_type == "ResNet18":
            print("==================================\nInitializing ResNet18\n==================================")
            model = ResNet18(
                num_channels = input_channels,
                num_classes = args.num_classes
            )
        elif args.model_type == "ResNet50" and args.dataset != "TinyImageNet": 
            print("==================================\nInitializing ResNet50\n==================================")
            model = ResNet50(
                num_channels = input_channels,
                num_classes = args.num_classes
            )
        elif args.model_type == "ResNet50" and args.dataset == "TinyImageNet":
            print("loading ResNet50 from huggingface")
            model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=200, ignore_mismatched_sizes=True )
            model = HuggingFaceModelWrapper(model)
        elif args.model_type == "ViT_B_16" or args.model_type == "ViT-B_16":
            print("==================================\nInitializing Pretrained ViT-B/16\n==================================")
            model = create_vit(model_name='vit_base_patch16_224', num_classes=args.num_classes)
            
        elif args.model_type == "ViT_S_16" or args.model_type == "ViT-S_16":
            print("==================================\nInitializing Pretrained ViT-S/16\n==================================")
            model = create_vit(model_name='vit_small_patch16_224', num_classes=args.num_classes)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        return model.to(self.device)

#add various models. first, 2L ReLU MLP

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layer=2):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        
        if num_layer == 2:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif num_layer > 2:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layer - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
    


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  
        return self.linear(x)
    

# ResNet Basic Block
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        # in case the input and output dimensions don't match
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
# ResNet Bottleneck block
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels

        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)
        # 3×3
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels)
        # 1×1 expand
        self.conv3 = nn.Conv2d(mid_channels, out_channels * Bottleneck.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * Bottleneck.expansion)

        # downsample if needed to match dimensions
        self.downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)

# ResNet Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        print(self.in_channels)
        print(block.expansion)
        self.linear = nn.Linear(self.in_channels, num_classes)

        # Apply (Gaussian) LeCun initialization
        # self._init_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        if num_blocks == 0:
            return nn.Sequential()  # empty layer (identity)

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        pool_size = out.size(-1)  # adaptive pooling
        out = F.avg_pool2d(out, pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

    """ 
    def _init_weights(self): # Gaussian LeCun initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Calculate fan_in: in_channels * kernel_height * kernel_width
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = 1 / (fan_in ** 0.5)  # beta = 1/sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0, std=std)
                # Note: Conv layers use bias=False, so no need to init bias
            elif isinstance(m, nn.Linear):
                fan_in = m.in_features
                std = 1 / (fan_in ** 0.5)  # beta_L = 1/sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # BatchNorm layers remain default (weight=1, bias=0)
    """ 

# ResNet18 model function
def ResNet18(num_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_channels, num_classes)
    
# ResNet50 model function
def ResNet50(num_channels=3, num_classes=10):
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_channels, num_classes)

def create_vit(model_name='vit_base_patch16_224', num_classes=200, img_size=224):
    model = timm.create_model(
        model_name, 
        pretrained=True,
        num_classes=num_classes,
        img_size=img_size,          
    )
    
    print(f"Loaded pretrained {model_name} with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model head output classes: {model.head.out_features}")
    
    return model


class HuggingFaceModelWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    
    def forward(self, x):
        outputs = self.model(x)
        # extract logits if they exist, otherwise return raw output
        return outputs.logits if hasattr(outputs, 'logits') else outputs
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)