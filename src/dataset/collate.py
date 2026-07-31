def collate_fn_torchvision(batch):
    imgs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return imgs, targets

