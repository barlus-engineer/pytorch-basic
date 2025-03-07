import torch
import torch.nn as nn
import torch.optim as optim

# dataset

x = torch.tensor([
    [1.0], [2.0], [3.0], [4.0], [5.0]
])
y = torch.tensor([
    [2.0], [4.0], [6.0], [8.0], [10.0]
])

# train

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
    
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")

model.eval()

# test model

with torch.no_grad():
    test = torch.tensor([7.])
    predict = model(test)
    print(f"predict for {test.item()} is {predict.item()}")