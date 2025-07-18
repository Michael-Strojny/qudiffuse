from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch
import torchvision

class Cutout:
    """Cutout augmentation for CIFAR-10."""
    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w)
        
        y = torch.randint(h, (1,))
        x = torch.randint(w, (1,))
        
        y1 = torch.clamp(y - self.size // 2, 0, h)
        y2 = torch.clamp(y + self.size // 2, 0, h)
        x1 = torch.clamp(x - self.size // 2, 0, w)
        x2 = torch.clamp(x + self.size // 2, 0, w)
        
        mask[y1:y2, x1:x2] = 0
        img = img * mask.unsqueeze(0)
        return img

class RescaleTransform:
    """Rescale normalized tensor to [0, 1] range."""
    def __call__(self, x):
        return (x + 1) / 2

def get_cifar10_transforms(train: bool = True, config: dict = None):
    """Get CIFAR-10 transforms with configurable augmentations."""
    if config is None:
        config = {}
    
    # Default augmentation settings
    aug_config = config.get("augmentation", {})
    horizontal_flip = aug_config.get("horizontal_flip", 0.5)
    random_crop = aug_config.get("random_crop", True)
    crop_size = aug_config.get("crop_size", 32)
    padding = aug_config.get("padding", 4)
    normalize = aug_config.get("normalize", True)
    cutout = aug_config.get("cutout", False)
    cutout_size = aug_config.get("cutout_size", 16)
    
    if train:
        transform_list = []
        
        # Random crop with padding
        if random_crop:
            transform_list.append(transforms.RandomCrop(crop_size, padding=padding))
        
        # Horizontal flip
        if horizontal_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if normalize:
            # Standard CIFAR-10 normalization to [0, 1] range for binary autoencoder
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                     (0.2023, 0.1994, 0.2010)))
            # Rescale to [0, 1] for binary autoencoder compatibility
            transform_list.append(RescaleTransform())
        
        # Cutout augmentation (applied after normalization)
        if cutout:
            transform_list.append(Cutout(size=cutout_size))
        
        transform = transforms.Compose(transform_list)
    else:
        transform_list = [transforms.ToTensor()]
        
        if normalize:
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                     (0.2023, 0.1994, 0.2010)))
            transform_list.append(RescaleTransform())
        
        transform = transforms.Compose(transform_list)
    
    return transform

def get_cifar10_loader(batch_size: int = 128, train: bool = True, download: bool = True, 
                      num_workers: int = 4, config: dict = None, target_class: int = None):
    """Get CIFAR-10 data loader with optimal transforms and settings."""
    
    if config is None:
        config = {}
    
    transform = get_cifar10_transforms(train=train, config=config)
    
    dataset = datasets.CIFAR10(
        root="./data", 
        train=train, 
        download=download, 
        transform=transform
    )
    
    # Filter by single class if specified (for unconditional training)
    if target_class is not None:
        # Get indices for the target class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
        dataset = Subset(dataset, class_indices)
        print(f"Filtered CIFAR-10 to class {target_class}: {len(class_indices)} samples")
    
    # Limit dataset size if specified (for quick testing)
    dataset_size_limit = config.get("dataset_size_limit", None)
    if dataset_size_limit is not None and dataset_size_limit < len(dataset):
        indices = list(range(dataset_size_limit))
        dataset = Subset(dataset, indices)
        print(f"Limited dataset to {dataset_size_limit} samples")
    
    # Get data loading settings from config
    pin_memory = config.get("pin_memory", torch.cuda.is_available())
    persistent_workers = config.get("persistent_workers", num_workers > 0)
    prefetch_factor = config.get("prefetch_factor", 2 if num_workers > 0 else None)
    
    # Prepare DataLoader kwargs
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": num_workers,
        "drop_last": train,
        "pin_memory": pin_memory,
    }
    
    # Only add these parameters if num_workers > 0
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    
    return DataLoader(**loader_kwargs) 

def get_cifar10_single_class_loaders(class_id: int = 0, batch_size: int = 128, 
                                   num_workers: int = 4, pin_memory: bool = True):
    """
    Create CIFAR-10 data loaders for single class training.
    
    Args:
        class_id: CIFAR-10 class ID (0-9)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Filter for single class
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_id]
    val_indices = [i for i, (_, label) in enumerate(val_dataset) if label == class_id]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    print(f"Single class '{classes[class_id]}' dataset loaded:")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")
    
    return train_loader, val_loader 