from torch.nn.parameter import Parameter


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchviz import make_dot

# NEURAL NETWORK
class NeuralNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # fc = fully connected - it's named this way because everyinput neuron is connected to every output neuron...it demonstrates the layers are "connected"
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         x = self.fc1(x) # pass x into the input of the first hidden layer
         x = F.relu(x) # Apply the ReLu activation function on fc1 neurons
         x = self.fc2(x) # pass the output of the ReLu activation function into fc2
         return x

# TRANSFORMER COMPONENTS
class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super.__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, max_seq_len: int=512, dropout: float=0.1):
        pass


if __name__ == "__main__":
    neural_network = NeuralNetwork(1,1, 1)
    x = torch.randn(1,1)
    output = neural_network(x)
    # print(output)

    dot = make_dot(output, params=dict[str, Parameter](neural_network.named_parameters()))
    dot.render("neural_network", format="png")  # Saves as neural_network.png

