import torch
import torch.nn.functional as F
from .FC import BayesianLinear


class BayesianMLP(torch.nn.Module):

    def __init__(self, args):
        super(BayesianMLP, self).__init__()

        ncha, size, _ = args.inputsize
        self.taskcla = args.taskcla
        self.samples = args.samples
        self.device = args.device
        self.sbatch = args.sbatch
        self.init_lr = args.lr
        dim = args.nhid
        self.nlayers = args.nlayers
        self.shared_head = getattr(args, 'shared_head', False)

        self.fc1 = BayesianLinear(ncha * size * size, dim, args)
        if self.nlayers == 2:
            self.fc2 = BayesianLinear(dim, dim, args)

        self.classifier = torch.nn.ModuleList()
        if self.shared_head:
            # Single shared output head across all domains (no task identity needed at test time)
            ncla = self.taskcla[0][1]
            self.classifier.append(BayesianLinear(dim, ncla, args))
        else:
            for t, n in self.taskcla:
                self.classifier.append(BayesianLinear(dim, n, args))

    def prune(self, mask_modules):
        for module, mask in mask_modules.items():
            module.prune_module(mask)

    def forward_features(self, x, sample=False, calculate_log_probs=None):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x, sample=sample, calculate_log_probs=calculate_log_probs))
        if self.nlayers == 2:
            x = F.relu(self.fc2(x, sample=sample, calculate_log_probs=calculate_log_probs))
        return x

    def forward_logits(self, x, t=0, sample=False, calculate_log_probs=False):
        """Fast forward returning raw logits for head t (or shared head 0) without redundant log_prior evaluations."""
        feat = self.forward_features(x, sample=sample, calculate_log_probs=calculate_log_probs)
        head_idx = 0 if self.shared_head else t
        return self.classifier[head_idx](feat, sample=sample, calculate_log_probs=calculate_log_probs)

    def kl_loss(self, t=0):
        """Compute KL(q || p) = (log_variational_posterior - log_prior) across active BayesianLinear layers."""
        kl = self.fc1.compute_kl()
        if self.nlayers == 2:
            kl = kl + self.fc2.compute_kl()
        head_idx = 0 if self.shared_head else t
        kl = kl + self.classifier[head_idx].compute_kl()
        return kl

    def forward(self, x, sample=False):
        x = self.forward_features(x, sample=sample, calculate_log_probs=None)
        if self.shared_head:
            logits = self.classifier[0](x, sample=sample, calculate_log_probs=None)
            log_probs = F.log_softmax(logits, dim=1)
            return [log_probs for _ in self.taskcla]
        y = []
        for t, _ in self.taskcla:
            y.append(self.classifier[t](x, sample=sample, calculate_log_probs=None))
        return [F.log_softmax(yy, dim=1) for yy in y]


def Net(args):
    return BayesianMLP(args)
