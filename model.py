import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from nlp_utils import tokenize, stem, bag_of_words

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.x_data = X
        self.y_data = y

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.x_data)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)

def train_model():
    with open('data/intents.json') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore = ['?', '!', '.', ',']
    all_words = sorted(set([stem(w) for w in all_words if w not in ignore]))
    tags = sorted(set(tags))

    X_train = []
    y_train = []
    for (pattern, tag) in xy:
        bow = bag_of_words(pattern, all_words)
        X_train.append(bow)
        y_train.append(tags.index(tag))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train)

    dataset = ChatDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = NeuralNet(len(all_words), 8, len(tags))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        for (words, labels) in loader:
            out = model(words)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save({
        "model": model.state_dict(),
        "input_size": len(all_words),
        "output_size": len(tags),
        "hidden_size": 8,
        "all_words": all_words,
        "tags": tags
    }, "data/chatbot.pth")