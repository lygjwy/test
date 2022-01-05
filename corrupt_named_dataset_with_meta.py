from torchvision.datasets.folder import default_loader
from named_dataset_with_meta import NamedDatasetWithMeta
from corrupt_transforms import GaussianNoise

class CorruptNamedDatasetWithMeta(NamedDatasetWithMeta):
    """load dataset with specified corruption

    Args:
        NamedDatasetWithMeta ([type]): [description]
    """
    def __init__(self, root, name, corruption, severity, split, transform, target_transform=None):
        super(CorruptNamedDatasetWithMeta, self).__init__(root=root, name=name, split=split, transform=transform, target_transform=target_transform)
        
        self.corruption = corruption
        self.severity = severity
        
        # get corruption & severity transform
        self.corruption = GaussianNoise(mean=0, std=0.1)
     
        
    def __getitem__(self, index):
        sample = self.samples[index]
        
        image_path = self.root / sample[0]
        image = default_loader(image_path)
        # corrupt original image
        image_corrupt = self.corruption(image)
        
        if self.transform is not None:
            image = self.transform(image)
            image_corrupt = self.transform(image_corrupt)
        
        if len(sample) == 1:
            return image_corrupt, image
        elif len(sample) == 2:
            target = sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return image_corrupt, image, target
        else:
            raise RuntimeError('---> wrong sample format')
        
    
    def __len__(self):
        return len(self.samples)
        
        
        
    