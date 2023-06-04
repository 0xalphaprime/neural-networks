import torch 
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device =  'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#_______________

torch.manual_seed(42)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the distinct characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from characters to indices and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: takes a list of integers, ouputs a string

# train and test the splits 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% train, 10% test
train_data, val_data = data[:n], data[n:]

# data loading 
def get_batch(split):
    # generate a small batch of data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def esitmate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# simple bigram model 
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) tensor of integers
        for _ in range(max_new_tokens):
            # get the predictions 
            logits, loss = self(idx)
            # focis only on the last time stamp
            logits = logits[:, -1, :]
            # apply the softmax to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# creat a PyTorch optimizer 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while eval the loss on train and val sets 
    if iter % eval_interval == 0:
        losses = esitmate_loss()
        print(f'iter = {iter}, train_loss = {losses["train"]:.4f}, val_loss = {losses["val"]:.4f}')
    
    # get a batch of data
    xb, yb = get_batch('train')

    # eval the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate some text from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))






