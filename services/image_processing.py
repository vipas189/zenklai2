import torchvision.transforms as transforms


def image_processing(img, data, validation=False, vit=False):
    x1, y1, x2, y2 = data[0], data[1], data[2], data[3]
    roi = img.crop((x1, y1, x2, y2))
    train_transform, val_transform, vit_train_transform, vit_val_transform = (
        transforming()
    )
    if validation:
        return vit_val_transform(roi) if vit else val_transform(roi)
    return vit_train_transform(roi) if vit else train_transform(roi)


def transforming():
    augmentations_64 = [
        # transforms.RandomRotation(15),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]

    augmentations_224 = [
        # transforms.RandomRotation(15),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]

    augmentation_probability = 0.75
    random_augment_64 = transforms.RandomApply(
        augmentations_64, p=augmentation_probability
    )
    random_augment_224 = transforms.RandomApply(
        augmentations_224, p=augmentation_probability
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            # random_augment_64,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    vit_train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # random_augment_224,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    vit_val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform, vit_train_transform, vit_val_transform
