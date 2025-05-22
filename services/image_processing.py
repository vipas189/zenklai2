import torchvision.transforms as transforms


def image_processing(img, data, validation=False, vit=False, augment=False):
    x1, y1, x2, y2 = data[0], data[1], data[2], data[3]
    roi = img.crop((x1, y1, x2, y2))
    train_transform, val_transform, vit_train_transform, vit_val_transform = (
        transforming()
    )
    if validation:
        return vit_val_transform(roi) if vit else val_transform(roi)
    else:
        if augment:
            return vit_train_transform(roi) if vit else train_transform(roi)
        else:
            return vit_val_transform(roi) if vit else val_transform(roi)


def transforming():
    train_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ]
    )

    vit_train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    vit_val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    return train_transform, val_transform, vit_train_transform, vit_val_transform
