import torch
import torch.nn as nn

class PerturbNet(torch.nn.Module):
    def __init__(self, 
                 n_input: int=148,
                 n_hiddens: int=2048,
                 dropout: float=0.5):
        super().__init__()
        
        self.n_input = n_input
        self.n_hiddens = n_hiddens
        self.dropout = dropout

        self.model = nn.Sequential(nn.Linear(self.n_input, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, 1))
        
    def forward(self, x):
        x = self.model(x)
        
        return x
    
class PerturbNet2(torch.nn.Module):
        def __init__(self, 
                     n_cell_type: int,
                     n_compount: int,
                     n_gene: int,
                 n_hiddens: int=2048,
                 dropout: float=0.5):
            super().__init__()
            
            self.n_cell_type = n_cell_type
            self.n_compount = n_compount
            self.n_gene = n_gene
            
            self.fc1 = nn.Linear(self.n_cell_type, 128)
            self.fc2 = nn.Linear(self.n_compount, 128)
            self.fc3 = nn.Linear(self.n_gene, 128)
            
            self.n_hiddens = n_hiddens
            self.dropout = dropout

            self.model = nn.Sequential(nn.Linear(128, self.n_hiddens),
                                    nn.BatchNorm1d(self.n_hiddens),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.n_hiddens, 1))

        def forward(self, x):
            x1 = self.fc1(x[:, :4])
            x2 = self.fc2(x[:, 4:20])
            x3 = self.fc3(x[:, 20:])
            
            x = x1 + x2 + x3
            x = self.model(x)
            
            return x
        
if __name__ == '__main__':
    model = PerturbNet2(4, 16, 128)
    x = torch.rand((10, 148))
    x = model(x)
    print(x)