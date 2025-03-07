import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

OPERATIONS = {
    "x+y": lambda x, y, p: (x, y, (x + y) % p),
    "x-y": lambda x, y, p: (x, y, (x - y) % p),
    "x*y": lambda x, y, p: (x, y, (x * y) % p),
    "x/y": lambda x, y, p: ((x*y) % p, y, x),
}

def create_dataset(operation:str, op_token:int, eq_token:int, prime:int):
    xs = torch.arange(0, prime)
    ys = torch.arange(int("/" in operation), prime)
    x,y = torch.cartesian_prod(xs, ys).T
    
    data = torch.zeros(len(x), 4, dtype=torch.long)
    x, y, labels = OPERATIONS[operation](x, y, prime)
    data[:,0] = x
    data[:,1] = op_token
    data[:,2] = y
    data[:,3] = eq_token
    return data, labels
    
def get_dataloaders(operation:str, op_token:int, eq_token:int, prime:int, 
                        batch_size=32, shuffle=True, train_fraction=0.8):
    data, labels = create_dataset(operation, op_token, eq_token, prime)
    dataset = torch.utils.data.TensorDataset(data, labels)
    train_size = int(train_fraction * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader

