import torch
import numpy as np
import json
import torch.nn as nn
import torch.optim as optim

# Load dataset
with open('data/motorbike-fuel-usage.json', 'r') as file:
    datas = json.load(file)

x, y = [], []

for data in datas:
    x.append([data['speed'], data['mass']])
    y.append([data['km_per_lit']])

x_tensor = torch.tensor(x).float()
y_tensor = torch.tensor(y).float()

class MotorbikeFuelUsage(nn.Module):
    def __init__(self):
        super(MotorbikeFuelUsage, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MotorbikeFuelUsage()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01)

epochs = 3000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print(f"epoch: {epoch+100}, loss: {loss.item()}")

model.eval()

with torch.no_grad():
    test = np.array([[62, 120]], dtype=np.float32)
    test_tensor = torch.tensor(test)
    predict = model(test_tensor).item()

    print(f"motorbike 62km/h, mass 120, fuel: {predict:.1f} km/l")
