import torch

# 1. Tensoren erstellen
scalar = torch.tensor(3.0)  # Skalar
vector = torch.tensor([1, 2, 3])  # Vektor
matrix = torch.tensor([[1, 2], [3, 4]])  # Matrix
tensor3d = torch.ones((3, 2, 2))

# 3. Tensoren wie NumPy-Arrays nutzen
matrix = matrix.float()  # Datentyp ändern
print(matrix @ matrix.T)  # Matrix-Multiplikation

# print
print(tensor3d)  # torch.Size([3, 2, 2])
print(tensor3d.device) # CPU oder GPU

###############################################
print("------------------------------------------------------------------------")

x = torch.tensor([2.0, 3.0], requires_grad=True)

y = torch.stack([x[0]**2, x[0]*x[1]])  # y0 = x0^2, y1 = x0*x1

jacobian = torch.zeros(2, 2)  # 2x2 Jacobian-Matrix
for i in range(len(y)):
    x.grad = None  # Gradienten zurücksetzen
    y[i].backward(retain_graph=True)
    jacobian[i] = x.grad  # Direkt in die i-te Zeile schreiben

    
print(jacobian)

