import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# DATASET
# =========================
data = [
    ("hello", "hi"),
    ("how are you", "i am fine"),
    ("what is your name", "i am chatbot"),
    ("bye", "goodbye")
]

# =========================
# PREPROCESSING
# =========================
def tokenize(sentence):
    return sentence.lower().split()

word2idx = {}
idx2word = {}
idx = 0

for inp, out in data:
    for word in tokenize(inp) + tokenize(out):
        if word not in word2idx:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1

vocab_size = len(word2idx)

def encode(sentence):
    return [word2idx[word] for word in tokenize(sentence)]

# =========================
# ATTENTION MODEL
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        encoder_outputs, (hidden, _) = self.encoder(embed)

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        output = self.fc(context.squeeze(1))
        return output

# =========================
# TRAINING
# =========================
model = ChatbotModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training started...\n")

for epoch in range(200):
    total_loss = 0

    for inp, out in data:
        input_seq = torch.tensor([encode(inp)])
        target_word = encode(out)[-1]   # ✅ better output (last word)

        output = model(input_seq)
        loss = criterion(output, torch.tensor([target_word]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("\nTraining completed!\n")

# =========================
# CHATBOT
# =========================
def chatbot():
    print("Chatbot is ready! (type 'exit' to stop)\n")

    while True:
        text = input("You: ").lower()

        if text == "exit":
            break

        words = tokenize(text)

        if any(word not in word2idx for word in words):
            print("Bot: I don't understand")
            continue

        input_seq = torch.tensor([encode(text)])
        output = model(input_seq)
        pred = torch.argmax(output).item()

        print("Bot:", idx2word[pred])

# Run chatbot
chatbot()