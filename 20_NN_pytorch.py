import torch
import torch.nn as nn
import torch.nn.functional as F

# Zielfunktion y=sin(x1​)+cos(x2​)

# Trainingsdaten erstellen
x = torch.rand(1000, 2) * 2 * torch.pi
y_true = torch.sin(x[:, 0:1]) + torch.cos(x[:, 1:2])

# Einfaches neuronales Netzwerk definieren
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# Modell, Verlustfunktion und Optimierer initialisieren
model = SimpleNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training
for epoch in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Testen des Modells
test = torch.tensor([[1.0, 2.0]])
print("\nTestinput:", test)
print("Vorhersage:", model(test).item())
print("Echter Wert:", (torch.sin(test[:,0]) + torch.cos(test[:,1])).item())