import torch
import argparse
import wandb
from data import get_dataloaders, OPERATIONS
from transformer import Transformer
from grokfast import gradfilter_ema
from tqdm import tqdm


def train(dataloader, optimizer, scheduler, loss_fn, model, device):
    model.train()
    grads = None
    for batch in dataloader:
        inputs, labels = batch                                          # (batch_size, context_len), (batch_size)
        optimizer.zero_grad()
        y_hat = model(inputs.to(device), pos=-1)                                            # (batch_size, vocab_size)
        loss = loss_fn(y_hat, labels.to(device))
        accuracy = (y_hat.argmax(dim=1) == labels.to(device)).float().mean()
        loss.backward()
        if args.use_filter:
            grads = gradfilter_ema(model, grads, aplha=args.alpha, lamb=args.lamb)
        optimizer.step()
        scheduler.step()
        
        wandb.log({
            "learning_rate": scheduler.get_last_lr()[0],
            "train/loss": loss,
            "train/accuracy": accuracy,
            "step": wandb.run.step})
        
        if wandb.run.step == args.num_steps:
            break


@torch.no_grad
def eval(dataloader, loss_fn, model, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    for batch in dataloader:
        inputs, labels = batch
        y_hat = model(inputs.to(device), pos=-1)
        total_loss += loss_fn(y_hat, labels.to(device)).item()
        total_accuracy += (y_hat.argmax(dim=1) == labels.to(device)).float().sum().item()
        
    loss = total_loss / len(dataloader.dataset)
    accuracy = total_accuracy / len(dataloader.dataset)
    return loss, accuracy


def init_logging(args):
    wandb.init(project="grok", config=args)
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("learning_rate",  step_metric="step")
    wandb.define_metric("train/loss",     step_metric="step")
    wandb.define_metric("train/accuracy", step_metric="step")
    wandb.define_metric("val/loss",       step_metric="epoch")
    wandb.define_metric("val/accuracy",   step_metric="epoch")


def main(args):
    torch.manual_seed(args.seed)
    init_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else \
        'mps' if torch.backends.mps.is_available() else 'cpu')

    # op and eq tokens are not necessary but are used for training
    op_token, eq_token = args.prime, args.prime + 1 # values >= prime to avoid collisions with numbers used for training
    train_loader, val_loader = get_dataloaders(args.operation, op_token, eq_token, args.prime, args.batch_size, train_fraction=args.train_fraction)
    model = Transformer(
        d_model=args.d_model,
        n_blocks=args.num_layers, 
        n_heads=args.num_heads,
        dropout=args.dropout,
        max_context_len=args.max_context_len,
        vocab_size=args.vocab_size,
        non_linearity=args.non_linearity).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10)
    
    num_epochs = int(args.num_steps / len(train_loader))
    for epoch in tqdm(range(num_epochs)):
        train(train_loader, optimizer, scheduler, criterion, model, device)
        loss, accuracy = eval(val_loader, criterion, model, device)
        wandb.log({
            "val/loss": loss,
            "val/accuracy": accuracy,
            "epoch": epoch})
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
    if args.save_weights:
        torch.save(model.state_dict(), f"{wandb.run.name}.pt")
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    # dataloader
    parser.add_argument("--operation", type=str, choices=OPERATIONS.keys(), default="x+y")
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=512)
    
    # model
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-context-len", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=100)
    parser.add_argument("--non-linearity", type=str, default="relu")
    
    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-steps", type=int, default=3e5)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--save-weights", action="store_true")
    
    # GrokFast
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--lamb", type=float, default=5.0)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)