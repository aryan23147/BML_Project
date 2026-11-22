import numpy as np
import torch
from torchvision import datasets


def get(data_path='../data/', seed=0, fixed_order=False, pc_valid=0.1, nperm=10, reverse=False):
    """
    Fast vectorized Permuted-MNIST dataloader:
      - 10 fixed pixel-permutation tasks (Task 0 is unpermuted identity when fixed_order=True).
      - Retains 10-class digit semantics across all tasks.
    """
    data = {}
    taskcla = []
    size = [1, 28, 28]

    seeds = list(range(nperm))
    if not fixed_order:
        rng = np.random.default_rng(seed)
        seeds = rng.permutation(seeds).tolist()
    if reverse:
        seeds = list(reversed(seeds))

    train_raw = datasets.MNIST(data_path, train=True, download=True)
    test_raw = datasets.MNIST(data_path, train=False, download=True)

    g = torch.Generator().manual_seed(seed)
    perm_split = torch.randperm(len(train_raw), generator=g)
    n_valid = int(len(train_raw) * pc_valid)
    valid_idx, train_idx = perm_split[:n_valid], perm_split[n_valid:]

    mean, std = 0.1307, 0.3081
    train_flat = ((train_raw.data.to(torch.float32) / 255.0 - mean) / std).view(len(train_raw), -1)
    train_targets = train_raw.targets.to(torch.long)

    test_flat = ((test_raw.data.to(torch.float32) / 255.0 - mean) / std).view(len(test_raw), -1)
    test_targets = test_raw.targets.to(torch.long)

    for t, r in enumerate(seeds):
        if r == 0:
            pix_perm = torch.arange(28 * 28, dtype=torch.long)
        else:
            g_perm = torch.Generator().manual_seed(int(r * 100 + 7))
            pix_perm = torch.randperm(28 * 28, generator=g_perm)

        data[t] = {
            'name': f'pmnist-{r}',
            'ncla': 10,
            'train': {
                'x': train_flat[train_idx][:, pix_perm].view(-1, 1, 28, 28).contiguous(),
                'y': train_targets[train_idx].clone(),
            },
            'valid': {
                'x': train_flat[valid_idx][:, pix_perm].view(-1, 1, 28, 28).contiguous(),
                'y': train_targets[valid_idx].clone(),
            },
            'test': {
                'x': test_flat[:, pix_perm].view(-1, 1, 28, 28).contiguous(),
                'y': test_targets.clone(),
            },
        }
        taskcla.append((t, 10))

    return data, taskcla, size
