import torch
import torch.nn.functional as F

# DAten
x = torch.linspace(-2, 2 , 10).unsqueeze(1)  # Eingabedaten
y_true = 2 * x + 1  # Wahre Ausgabedaten mit Rauschen

# Parameter
# Input → Hidden (1 Input, 2 Hidden)
w1 = torch.randn(1,2, requires_grad=True)  # Gewicht 1
b1 = torch.randn(2, requires_grad=True)    # Bias 1
# Hidden → Output (2 Hidden, 1 Output)
w2 = torch.randn(2,1, requires_grad=True)  # Gewicht 2
b2 = torch.randn(1, requires_grad=True)    # Bias 2

# Training
learning_rate = 0.01

for epoch in range(1000):
    # Vorwärtsdurchlauf
    h = torch.matmul(x, w1) + b1  # Lineare Transformation
    h_act = torch.tanh(h)        # Aktivierungsfunktion
    y_pred = torch.matmul(h_act, w2) + b2  # Ausgabevorhersage

    # Verlust berechnen (Mean Squared Error)
    loss = F.mse_loss(y_pred, y_true)
    # Rückwärtsdurchlauf
    loss.backward()

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

    # Alle 100 Epochen ausgeben
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")