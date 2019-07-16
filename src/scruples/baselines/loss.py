"""Custom losses for baselines."""

import torch as th
from torch.nn import functional as F


class SoftCrossEntropyLoss(th.nn.Module):
    """Cross-entropy with soft reference labels."""
    # N.B. This loss can be used in two ways. First, labels can be
    # averaged and then passed in as the targets, and second, the raw
    # counts of each label can be passed in as the target. The first
    # approach leverages the soft labels, but ignores relative certainty
    # of one instances labeling versus another. The second approach
    # leverages all of the annotation information available. Both are
    # equivalent if each instance has the same number of labels.

    def forward(self, input, target):
        return - th.mean(th.sum(target * F.log_softmax(input), dim=-1))


class DirichletMultinomialLoss(th.nn.Module):
    """Negative log-likelihood for a dirichlet-multinomial."""

    # N.B. note that this function computes the likelihood of the
    # observed labels, and not the likelihood of the sufficient
    # statistic derived from them (i.e., the counts of each label). We
    # only need the sufficient statistic however to compute this
    # likelihood, and both lead to the same MLE.
    def forward(self, inputs, targets):
        inputs = th.exp(inputs)
        return - th.mean(
            th.lgamma(th.sum(inputs, dim=-1))
            + th.sum(th.lgamma(inputs + targets), dim=-1)
            - th.lgamma(th.sum(inputs + targets, dim=-1))
            - th.sum(th.lgamma(inputs), dim=-1))
