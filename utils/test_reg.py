import torch
import matplotlib.pyplot as plt
import torch.nn as nn
# Définir le modèle de régression non linéaire
class NonLinearRegression(nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()
        self.layer1 = nn.Linear(1, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.tanh(self.layer3(out))
        return out

# Instancier le modèle et définir la fonction de perte et l'optimiseur
# model 1
model = NonLinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# model 2
model2 = NonLinearRegression()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

# Créer un dataset artificiel
xtrain = torch.linspace(-1, 1, 10).unsqueeze(1)
ytrain = 0.1*torch.randn(10, 1)
epsilon=0.005
# old prediction
xeval = torch.linspace(-1, 1, 100).unsqueeze(1)
old_predicted = model(xeval).detach().numpy()

# Entraîner le modèle
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(xtrain)
    # custom loss
    x_0= xtrain+ epsilon*torch.randn(10, 1)
    x_0.requires_grad=True
    dist_max=criterion(model(x_0),model(xtrain))
    grads = torch.autograd.grad(dist_max, x_0, create_graph=True)[0]
    perturbation = 1/(epsilon)*grads.data
    x_tilde=x_0+perturbation
    x_tilde=x_tilde.detach()
    custom_loss = criterion(model(x_tilde), model(xtrain))
    # print(custom_loss)
    # # standard loss
    formal_loss = criterion(outputs, ytrain)
    # # final loss
    loss=10*custom_loss+formal_loss

    # Backward pass et mise à jour des poids
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # model 2 train 
    outputs2 = model2(xtrain)
    # formal loss
    formal_loss2 = criterion(outputs2, ytrain)
    optimizer2.zero_grad()
    formal_loss2.backward()
    optimizer2.step()
    # Afficher le loss
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:}'.format(epoch+1, num_epochs, loss.item()))

# Afficher la prédiction du modèle
predicted = model(xeval).detach().numpy()
predicted2 = model2(xeval).detach().numpy()
plt.plot(xeval.detach().numpy(), predicted2, 'y')
plt.plot(xeval.detach().numpy(), predicted, 'r')
plt.plot(xeval.detach().numpy(), old_predicted, 'b')
plt.scatter(xtrain.detach().numpy(), ytrain.numpy(), c='g')
plt.show()
