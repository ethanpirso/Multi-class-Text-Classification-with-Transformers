import argparse
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dataloader import create_data_loaders
from models import TransformerClassifier

def main(args):
    train_loader, val_loader, test_loader, num_classes = create_data_loaders(args.batch_size, args.max_len)

    model = TransformerClassifier(num_classes=num_classes, learning_rate=args.learning_rate)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best_model',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-class Text Classification using Transformers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum length of tokenized input sequences')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=7, help='Number of training epochs')
    args = parser.parse_args()
    main(args)
