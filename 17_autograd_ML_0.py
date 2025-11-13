import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])  # ideale Gerade: y = 2x + 1

# parameter
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.1

for epoch in range(100):
    y_pred = w * x + b
    loss = torch.mean((y_true - y_pred) ** 2)
    loss.backward()
    # Update der Parameter
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    # print
    print(f"Epoch {epoch+1}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

