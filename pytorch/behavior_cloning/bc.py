import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BC(nn.Module):
    def __init__(self, input_size=77):
        output_size = 6
        super(BC, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        prob = F.softmax(x, dim=0)
        return prob


    def train_net(self, data):
        s, a = data

        # new_s = [] # eliminate noop
        # new_a = []

        # for e, f in zip(s, a):
        #     if f == 0:
        #         pass
        #     else:
        #         new_s.append(e)
        #         new_a.append(f)

        #s, a = torch.tensor(new_s, dtype=torch.float), torch.tensor(new_a, dtype=torch.long)
        s, a = torch.tensor(s, dtype=torch.float), torch.tensor(a, dtype=torch.long)
        
        probs = self.forward(s)
        loss = F.cross_entropy(probs, a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


def get_batch(np_file, batch_size=32):
    for e in range((len(np_file["states"])//batch_size)-1):
        yield np_file["states"][e*batch_size:(e+1)*batch_size], np_file["actions"][e*batch_size:(e+1)*batch_size]
    

def train():
    model = BC()
    np_file = np.load("food.npz")
    loss = 0
    
    epoch = 1000000

    for i in range(epoch):
        for data in get_batch(np_file):
            loss = model.train_net(data)
        if i % 100 == 0:
            print(f"{i} : {loss}")
            
        if i % 1000 == 0 and i != 0:
            torch.save(model.state_dict(), f"imitation_model_{i}.pt")
            print("saved!")


if __name__ == "__main__":
    train()