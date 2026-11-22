import torch


class BalancedReplayBuffer:
    """
    Fixed-capacity replay buffer balanced evenly across observed (task_id, class_id) buckets.
    Default budget: capacity = 200 examples total.
    """

    def __init__(self, capacity=200, device='cpu'):
        self.capacity = int(capacity)
        self.device = device
        self.storage = {}  # (task_id, class_id) -> (x_cpu, y_cpu)

    def is_empty(self):
        return len(self.storage) == 0

    def __len__(self):
        return sum(bx.size(0) for bx, _ in self.storage.values())

    def update_after_task(self, task_id, x_train, y_train, seed=0):
        g = torch.Generator().manual_seed(int(seed) + int(task_id))
        x_cpu = x_train.detach().cpu()
        y_cpu = y_train.detach().cpu()
        classes = sorted(torch.unique(y_cpu).tolist())

        new_buckets = {}
        for c in classes:
            idx = (y_cpu == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(len(idx), generator=g)]
            new_buckets[(int(task_id), int(c))] = (x_cpu[perm], y_cpu[perm])

        total_buckets = len(self.storage) + len(new_buckets)
        per_bucket = max(1, self.capacity // total_buckets)
        remainder = max(0, self.capacity - per_bucket * total_buckets)

        updated = {}
        for i, (key, (bx, by)) in enumerate(self.storage.items()):
            quota = per_bucket + (1 if i < remainder else 0)
            updated[key] = (bx[:quota].clone(), by[:quota].clone())

        offset = len(self.storage)
        for j, (key, (bx, by)) in enumerate(new_buckets.items()):
            quota = per_bucket + (1 if (offset + j) < remainder else 0)
            updated[key] = (bx[:quota].clone(), by[:quota].clone())

        self.storage = updated

    def get_all(self):
        xs, ys = [], []
        for bx, by in self.storage.values():
            xs.append(bx)
            ys.append(by)
        return torch.cat(xs, dim=0).to(self.device), torch.cat(ys, dim=0).to(self.device)
