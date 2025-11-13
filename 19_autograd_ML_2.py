import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Zielfunktion y=sin(x1​)+cos(x2​)
# Trainingsdaten generieren
x = torch.rand(100, 2) * 2 * torch.pi  # Zufällige Werte für x1 und x2 im Bereich [0, 2π]
y = torch.sin(x[:, 0:1]) + torch.cos(x[:, 1:2])  # Zielwerte berechnen

# Parameter für das lineare Modell 2 Input, 3 Hidden, 1 Output
w1 = torch.randn(2, 3, requires_grad=True)  # Gewichte für Eingabe zu versteckter Schicht
b1 = torch.randn(3, requires_grad=True)     # Bias für versteckte Schicht
w2 = torch.randn(3, 1, requires_grad=True)  # Gewichte für versteckte zu Ausgabeschicht
b2 = torch.randn(1, requires_grad=True)     # Bias für Ausgabeschicht
learning_rate = 0.01

# Training
for epoch in range(1000):
    # Vorwärtsdurchlauf
    hidden = torch.matmul(x, w1) + b1
    hidden_activated = F.tanh(hidden)  # Aktivierungsfunktion
    y_pred = torch.matmul(hidden_activated, w2) + b2

    # Verlust berechnen (Mean Squared Error)
    Loss = F.mse_loss(y_pred, y)

    # Rückwärtsdurchlauf
    Loss.backward()

    # Parameter aktualisieren
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad

        # Gradienten zurücksetzen
        w1.grad.zero_()
        b1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()

    # if epoch % 100 == 0:
    #     print(f'Epoch {epoch}, Loss: {Loss.item()}')


#################### Testen des Modells

# Testdaten generieren
x_test = torch.rand(100, 2) * 2 * torch.pi
y_true_test = torch.sin(x_test[:, 0:1]) + torch.cos(x_test[:, 1:2])

# Modell testen
with torch.no_grad():
    hidden_test = torch.matmul(x_test, w1) + b1
    hidden_activated_test = F.tanh(hidden_test)
    y_pred_test = torch.matmul(hidden_activated_test, w2) + b2

# Metriken berechnen
test_loss = F.mse_loss(y_pred_test, y_true_test)
ss_total = torch.sum((y_true_test - torch.mean(y_true_test)) ** 2)
ss_residual = torch.sum((y_true_test - y_pred_test) ** 2)
r_squared = 1 - (ss_residual / ss_total)
mae = torch.mean(torch.abs(y_true_test - y_pred_test))

# Ergebnisse ausgeben
print(f"Test-Loss (MSE): {test_loss.item():.6f}")
print(f"R²-Score: {r_squared.item():.6f}")
print(f"Mean Absolute Error (MAE): {mae.item():.6f}")