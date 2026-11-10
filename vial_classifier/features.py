from torchvision import transforms
from vial_classifier.config import IMAGE_SIZE, TRANSFORM_MEAN, TRANSFORM_STD


def define_transforms():
    print("Defining image transforms...")

    # Training transforms with data augmentation
    train_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
        ]
    )

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
        ]
    )

    print("Training transforms: Resize, flip, rotation, jitter, ToTensor, Normalize")
    print("Validation transforms: Resize, ToTensor, Normalize")
    return train_transforms, val_transforms