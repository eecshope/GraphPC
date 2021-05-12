import torch
from typing import Any
from pytorch_lightning.metrics import Metric


class ACC(Metric):
    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass

    def __init__(self, dist_sync_on_step=False):
        super(ACC, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _get_acc(logits: torch.Tensor, target: torch.Tensor):
        n_vocab = logits.shape[-1]
        logits = logits.detach().reshape((-1, n_vocab))  # [-1, vocab]
        target = target.detach().reshape((-1))  # [-1]
        preds = torch.argmax(logits, -1)  # [-1]
        # hits = torch.mul(torch.eq(target, preds), torch.ne(target, 0)).sum()
        hits = torch.sum(torch.eq(target, preds))

        return hits

    def update(self, *args):
        logits, target = args
        self.acc += ACC._get_acc(logits, target)
        self.total += target.numel()

    def compute(self):
        return self.acc / self.total


class AverageMetric(Metric):
    def __init__(self):
        super(AverageMetric, self).__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, *args):
        loss, num = args
        self.sum += num * loss
        self.total += num

    def compute(self):
        return self.sum / self.total
