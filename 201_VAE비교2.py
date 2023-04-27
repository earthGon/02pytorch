import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, dim_latent = 2):
        super().__init__()

        self.dim_latent = dim_latent

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, dim_latent * 2),# latent space dimension의 2배
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(dim_latent, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28*28),
        )

    def forward(self, x):
        x = self.encoder(x).view(-1, 2, self.dim_latent)

        # reparameterize
        self.mu = x[:, 0, :]
        self.sigma = torch.exp(x[:, 1, :])
        self.z = self.mu + self.sigma * torch.randn_like(self.mu)
        return self.decoder(self.z)

model = VariationalAutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data_loader = DataLoader(MyDataset(), batch_size=1000, shuffle=True)

for epoch in range(1, 101):
    epoch_loss = 0.0
    for x_input in data_loader:
        optimizer.zero_grad()
        x_hat = model(x_input)

        mse_loss = ((x_input - x_hat)**2).sum()
        kl_loss = (model.sigma**2 + model.mu**2 - torch.log(model.sigma) - 0.5).sum()
        loss = mse_loss + kl_loss

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item()}")