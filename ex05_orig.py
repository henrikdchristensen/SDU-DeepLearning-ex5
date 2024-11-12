import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_key):
        super().__init__()
        # Three separate linear layers for the queries, keys, and values
        self.w_q = nn.Linear(d_model, d_key)
        self.w_k = nn.Linear(d_model, d_key)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # ... your code here ...
        pass
        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_model, d_key) for _ in range(n_heads)])
        # Down projection back to model dimension
        # Alternatively, we could also split the input into n_heads and concatenate the output
        self.w_o = nn.Linear(n_heads * d_model, d_model)

    def forward(self, x):
        # ... your code here ...
        pass

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_key, n_heads, mlp_factor=4):
        super().__init__()
        # We need to init two layer norms because they have parameters
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_key, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        # a feedforward module with one internal hidden layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_factor * d_model),
            nn.SiLU(),  # Swish activation function, f(x) = x * sigmoid(x)
            nn.Linear(mlp_factor * d_model, d_model)
        )

    def forward(self, x):
        # ... your code here ...
        pass

class TransformerClassifier(nn.Module):
    def __init__(self, n_embeds, n_classes, d_model=256, d_key=64, n_heads=4, mlp_factor=4, n_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(n_embeds, d_model)
        self.transformer_model = nn.Sequential(*[TransformerBlock(d_model, d_key, n_heads, mlp_factor) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_classes))

    def forward(self, x):
        # ... your code here ...
        pass



# --------- #

if __name__ == '__main__':
    # Even odd task
    # Create a dataset of even and odd numbers
    # Set seeds
    torch.manual_seed(0)

    x = torch.rand(1000, 10) < 0.5
    y = x.sum(dim=1) % 2
    x = x.long()

    # Split the dataset into training and validation sets
    x_train, x_val = x[:800], x[800:]
    y_train, y_val = y[:800], y[800:]

    # Create the model
    model = TransformerClassifier(2, 2, d_model=8, d_key=8, n_heads=2, mlp_factor=4, n_layers=2)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    # Check the validation accuracy
    with torch.no_grad():
        model.eval()
        y_pred = model(x_val)
        acc = (torch.argmax(y_pred, dim=1) == y_val).float().mean()
        print(f'Validation accuracy: {100*acc.item()}%')

    # Number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

# --------- #
# Test code #
# --------- #

# Run tests via `pytest ex05.py`
# The tests only test the shapes of the outputs, not the actual values

def test_SelfAttention():
    # Test the Attention module
    att = SelfAttention(64, 16)
    x = torch.randn(32, 10, 64)
    z = att(x)
    assert ((z - x).abs() > 1e-6).any()
    assert z.shape == (32, 10, 64), z.shape

def test_MultiHeadAttention():
    # Test the MultiHeadAttention module
    mha = MultiHeadSelfAttention(64, 16, 8)
    x = torch.randn(32, 10, 64)
    z = mha(x)
    assert ((z - x).abs() > 1e-6).any()
    assert z.shape == (32, 10, 64), z.shape

def test_TransformerBlock():
    # Test the TransformerBlock module
    tb = TransformerBlock(64, 16, 8)
    x = torch.randn(32, 10, 64)
    z = tb(x)
    assert ((z - x).abs() > 1e-6).any()
    assert z.shape == (32, 10, 64), z.shape

def test_TransformerClassifier():
    # Test the TransformerClassifier module
    t = TransformerClassifier(2, 2)
    x = torch.randint(2, (32, 10))
    z = t(x)
    assert z.shape == (32, 2), z.shape