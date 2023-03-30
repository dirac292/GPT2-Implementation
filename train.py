
import torch
import torch.nn as nn
from torch.nn import functional as F


#Hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # 384/6 = 64 every head is 64 dimensional
n_head = 6
n_layer = 6
dropout = 0.2
# -----------------------------------------------
torch.manual_seed(1337)

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenize the input text that convert raw text (Chars to integers)

stoi = {ch: i for i,ch in enumerate(chars)}
print(stoi)
itos = {i: ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]# Encoder takes a string,output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Takes a list of strings and converts it to a string

data = torch.tensor(encode(text),dtype = torch.long)
# Train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

@torch.no_grad()
def estimate_loss(): # Averages the loss 
    out = {}
    model.eval() # Setting model to evaluation phase
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Setting the model to training phase
    return out

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias = False)
        self.query = nn.Linear(n_embd,head_size,bias = False)
        self.value = nn.Linear(n_embd,head_size,bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # B T C
        q = self.query(x) # B T C

        wei = q @ k.transpose(-2,-1) * C**-0.5 # B T T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float ('-inf'))
        wei = F.softmax(wei,dim = -1)
        wei = self.dropout(wei) # Shuts off some subset of neurons and trains without them
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)
        out = self.dropout(self.proj(out))
        return out 
# batch_size = 4 # how many indep sequ we process in parallel
# block_size = 8 

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size,(batch_size,)) # batch size number of offsets so 4 offsets of size (len(data) - block_size)
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y = x.to(device),y.to(device)
  return x,y
xb,yb = get_batch('train')

class LayerNorm(nn.Module):
  def __init__(self,dim,eps=1e-5,momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    
  def __call__(self,x):
    # calc the forfward pass
    dim = 1
    xmean = x.mean(dim,keepdim = True)
    xvar = x.var(dim,keepdim = True)
    xhat = (x - xmean)/ torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    
    return self.out
  
  def parameters(self):
    return [self.gamma,self.beta]


class FeedForward(nn.Module):
    """ Linear layer followed by non linearity"""
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd) # N_embd number of embedding suggestions
    self.position_embedding_table = nn.Embedding(block_size,n_embd) # each position 0 to block_size - 1 gets an embedding
    self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
    # self.blocks = nn.Sequential( # embd = 32 n_head = 4 and head_size = 8
    #     Block(n_embd,n_head = 4),
    #     Block(n_embd,n_head = 4),
    #     Block(n_embd,n_head = 4),
    #     nn.LayerNorm(n_embd)
    # )
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd,vocab_size)

  def forward(self,idx,targets = None):
    B,T = idx.shape
    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) #(B,T,C) -> 4 x 8 x 65
    pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # T,C
    x = tok_emb + pos_emb
    x = self.blocks(x)
    logits = self.lm_head(x) # B,T,vocab_size

    if targets is None :
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C) 
      targets = targets.view(B*T) # B x T
      loss = F.cross_entropy(logits,targets) # need B C T and not B T C

    return logits,loss

  def generate(self,idx,max_new_tokens): # Make 
    # idx B,T array of indices in the current context
    for _  in range(max_new_tokens):
      #  crop idx to last block_size tokens
      idx_cond = idx[:,-block_size:]
      # get the predictions
      logits,loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[: , -1, :] # becomes (B,C)
      # apply softmax to get probabilities
      probs = F.softmax(logits,dim = 1) # B C
      # sample from the dist
      idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
      # append sampled index to the running sequence
      idx = torch.cat((idx,idx_next),dim = 1) # B, T + 1
    return idx

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)


for iter in range(max_iters):
  # Every once in a while evaluate the loss on train and val sets
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f} val loss{losses['val']:.4f}")

  # sample a batch of data
  xb ,yb = get_batch('train')
  # evaluate the loss
  logits,loss = m(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1,1),dtype = torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))