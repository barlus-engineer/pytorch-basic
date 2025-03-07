import torch
import numpy as np
import json
import torch.nn as nn
import torch.optim as optim

# dataset
with open('data/motorbike-fuel-usage.json', 'r') as file:
    datas = json.load(file)

x, y = [], []

for data in datas:
    x.append([data['speed'], data['mass']])
    y.append([data['km_per_lit']])

x_tensor = torch.tensor(x)
y_tensor = torch.tensor(y)

class ANNPredictNextNumber(nn.Module):
    def __init__(self):
        super(ANNPredictNextNumber, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = ANNPredictNextNumber()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor.float())
    loss = criterion(outputs, y_tensor.float()) 
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss}")

model.eval()

with torch.no_grad():
    test = np.array([50, 120])
    test_tensor = torch.from_numpy(test)
    predict = model(test_tensor.float()).item()
    print("Predict:", predict)
