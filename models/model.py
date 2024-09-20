import torch
import torch.nn

class InputEmbeddings(torch.nn.Module):

    def __init__(self,d_model,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size,d_model)
    
    def forword(self,x):
        return self.embedding(x)

