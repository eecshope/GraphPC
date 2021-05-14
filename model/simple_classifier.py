import torch
import pytorch_lightning

from torch import nn
from torch.optim import Adam
from dgl.nn.pytorch.conv import GatedGraphConv
from dgl.nn.pytorch import GlobalAttentionPooling
from torch.nn.functional import cross_entropy
from utils.metrics import ACC, AverageMetric
from graph_construct import cpp_parser_


class SimpleClassifier(pytorch_lightning.LightningModule):
    def __init__(self, vocab_size, n_features, n_classes, n_layers):
        super(SimpleClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_layers = n_layers

        # allocate the tensors
        self.embed = nn.Embedding(vocab_size, n_features)
        self.backbone = GatedGraphConv(n_features, n_features, n_layers, cpp_parser_.EDGE_KIND * 2)
        pooling_gate_nn = nn.Linear(n_features, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn)
        self.linear_cls = nn.Linear(n_features, n_classes)

        # metrics
        self.acc = ACC()
        self.avg_loss = AverageMetric()

    def forward(self, graph):
        node_idx = graph.ndata["idx"].long()
        word_emb = self.embed(node_idx)
        edge_type = graph.edata["type"]

        h = self.backbone(graph, word_emb, edge_type)
        logits = self.linear_cls(self.pooling(graph, h))

        return logits

    def training_step(self, batch, batch_idx):
        graphs, labels = batch
        logits = self.forward(graphs)
        loss = cross_entropy(logits, labels.long())
        self.log("train_loss", loss.detach())
        return loss

    def _valid_test_step(self, graphs, labels):
        with torch.no_grad():
            logits = self.forward(graphs)
            loss = cross_entropy(logits, labels.long())
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
        return Adam(self.parameters(), lr=0.0001)
