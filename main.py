import argparse
import pytorch_lightning as pl
from utils.data import ASTDataset
from model.simple_classifier import SimpleClassifier
from dgl.dataloading import GraphDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ProgramData")
    parser.add_argument("--root_dir", default="saved", help="Root Directory for Logs and Models")
    parser.add_argument("--task", help="The task for the call, either 'train' or 'test' ")
    parser.add_argument("--save_path", default=None, help="Path of the archived model weights")

    parser.add_argument("--vocab_size", default=7961, help="size of the vocabulary")
    parser.add_argument("--n_features", default=256, help="dimension of the features")
    parser.add_argument("--n_classes", default=10, help="number of classes")
    parser.add_argument("--n_layers", default=5, help="number of stacked layers")
    args = parser.parse_args()

    if args.task not in ["train", "test"]:
        raise ValueError(f"Task {args.task} is unknown. Task should be within ['train', 'test']")

    print("Building Model...")
    model = SimpleClassifier(vocab_size=args.vocab_size,
                             n_features=args.n_features,
                             n_classes=args.n_classes,
                             n_layers=args.n_layers)
    if args.task == "test":
        print(f"Loading model from checkpoint {args.save_path}")
        model.load_from_checkpoint(args.save_path)

    print(f"Loading {args.task} data from {args.data_dir}...")
    if args.task == "test":
        test_dataset = ASTDataset("test", save_dir=args.data_dir)
    else:
        train_dataset = ASTDataset("train", save_dir=args.data_dir)
        valid_dataset = ASTDataset("valid", save_dir=args.data_dir)

    print("Building Trainer...")
    trainer = pl.Trainer(gpus=1, max_epochs=10, terminate_on_nan=True, default_root_dir=args.root_dir)

    if args.task == "test":
        print("Test not ready")
    else:
        trainer.fit(model, train_dataloader=GraphDataLoader(train_dataset, batch_size=32),
                    val_dataloaders=GraphDataLoader(valid_dataset, batch_size=32))


if __name__ == "__main__":
    main()