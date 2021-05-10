import dgl
import torch
import pytorch_lightning

from torch import nn
from torch.optim import Adam
from dgl.nn.pytorch.conv import GraphConv
from torch.nn.functional import cross_entropy
from utils.metrics import ACC, AverageMetric


class SimpleClassifier(pytorch_lightning.LightningModule):
    def __init__(self, vocab_size, n_features, n_classes, n_layers):
        super(SimpleClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_layers = n_layers

        # allocate the tensors
        self.embed = nn.Embedding(vocab_size, n_features)
        self.backbones = [[GraphConv(n_features, n_features), nn.ReLU()] for _ in range(n_layers)]
        self.final_project = GraphConv(n_features, n_classes)

        # metrics
        self.acc = ACC()
        self.avg_loss = AverageMetric()

    def forward(self, graph):
        node_idx = graph.ndata["idx"].long()
        word_emb = self.embed(node_idx)
        h = word_emb

        for i in range(self.n_layers):
            h = self.backbones[i][0](graph, h)
            h = self.backbones[i][1](h)

        graph.ndata["h"] = h
        return dgl.mean_nodes(graph, "h")

    def training_step(self, batch, batch_idx):
        graphs, labels = batch
        logits = self.forward(graphs)
        loss = cross_entropy(logits, labels)
        self.log("train_loss", loss.detach())
        return loss

    def _valid_test_step(self, graphs, labels):
        with torch.no_grad():
            logits = self.forward(graphs)
            loss = cross_entropy(logits, labels)
            self.acc(logits.detach(), labels.detach())
            self.avg_loss(loss.detach(), labels.shape[0])

    def validation_step(self, batch, *args, **kwargs):
        graphs, labels = batch
        self._valid_test_step(graphs, labels)
        self.log("val_loss", self.avg_loss)
        self.log("val_acc", self.acc)

    def test_step(self, batch, *args, **kwargs):
        graphs, labels = batch
        self._valid_test_step(graphs, labels)
        self.log("test_loss", self.avg_loss)
        self.log("test_acc", self.acc)

    def configure_optimizers(self):
        return Adam(self.parameters())
