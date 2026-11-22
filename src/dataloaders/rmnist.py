import torch
from torchvision import datasets
import torchvision.transforms.functional as TF


def get(data_path='../data/', seed=0, fixed_order=False, pc_valid=0.1, angle_gap=10, reverse=False, num_tasks=5):
    """
    Rotated-MNIST dataloader for Idea 2:
      - 5 domains starting at 0 deg: (0, 10, 20, 30, 40) or (0, 30, 60, 90, 120)
      - Supports forward and reverse task orders while retaining digit-label semantics (10 classes).
      - Uses batched tensor rotation so all 5 domains construct in < 1s on CPU.
    """
    data = {}
    taskcla = []
    size = [1, 28, 28]

    angles = [i * angle_gap for i in range(num_tasks)]
    if reverse:
        angles = list(reversed(angles))

    train_raw = datasets.MNIST(data_path, train=True, download=True)
    test_raw = datasets.MNIST(data_path, train=False, download=True)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_raw), generator=g)
    n_valid = int(len(train_raw) * pc_valid)
    valid_idx, train_idx = perm[:n_valid], perm[n_valid:]

    mean, std = 0.1307, 0.3081

    # Convert raw uint8 [N, 28, 28] to [N, 1, 28, 28] in [0, 1] once
    train_imgs_01 = train_raw.data.unsqueeze(1).to(torch.float32) / 255.0
    train_targets = train_raw.targets.to(torch.long)

    test_imgs_01 = test_raw.data.unsqueeze(1).to(torch.float32) / 255.0
    test_targets = test_raw.targets.to(torch.long)

    def build_split(imgs_01, targets, indices, angle):
        sub_imgs = imgs_01[indices]
        sub_targets = targets[indices].clone()
        if angle != 0:
            sub_imgs = TF.rotate(sub_imgs, float(angle))
        sub_imgs = (sub_imgs - mean) / std
        return sub_imgs, sub_targets

    for t, angle in enumerate(angles):
        data[t] = {
            'name': f'rmnist-{angle}deg',
            'angle': angle,
            'ncla': 10,
            'train': {},
            'valid': {},
            'test': {},
        }
        data[t]['train']['x'], data[t]['train']['y'] = build_split(
            train_imgs_01, train_targets, train_idx, angle
        )
        data[t]['valid']['x'], data[t]['valid']['y'] = build_split(
            train_imgs_01, train_targets, valid_idx, angle
        )
        data[t]['test']['x'], data[t]['test']['y'] = build_split(
            test_imgs_01, test_targets, torch.arange(len(test_raw)), angle
        )
        taskcla.append((t, 10))

    return data, taskcla, size
