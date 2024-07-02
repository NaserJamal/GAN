import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from google.colab import files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Generator(nn.Module):
    def __init__(self, latent_dim, state_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 4, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, state_dim),
            nn.Tanh()
        )

    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, state_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + 4, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

class SnakeGAN:
    def __init__(self, state_dim, latent_dim=50, lr=0.0002, beta1=0.5):
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.generator = Generator(latent_dim, state_dim).to(device)
        self.discriminator = Discriminator(state_dim).to(device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        self.criterion = nn.BCELoss()

    def train_step(self, real_states, actions):
        batch_size = real_states.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        self.d_optimizer.zero_grad()

        real_outputs = self.discriminator(real_states, actions)
        d_loss_real = self.criterion(real_outputs, real_labels)

        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_states = self.generator(z, actions)
        fake_outputs = self.discriminator(fake_states.detach(), actions)
        d_loss_fake = self.criterion(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        self.g_optimizer.zero_grad()
        fake_outputs = self.discriminator(fake_states, actions)
        g_loss = self.criterion(fake_outputs, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

def preprocess_data(states, actions):
    states = states.astype(np.float32) / 255.0
    states = states.reshape(states.shape[0], -1)

    actions_one_hot = np.zeros((actions.shape[0], 4), dtype=np.float32)
    actions_one_hot[np.arange(actions.shape[0]), actions] = 1

    return states, actions_one_hot

# def estimate_memory_usage(model, input_size, batch_size):
#     input = torch.randn(batch_size, *input_size).to(device)
#     torch.cuda.reset_peak_memory_stats()
#     _ = model(input)
#     memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert to MB
#     return memory_usage

def train(gan, states, actions, num_epochs=10, batch_size=4, accumulation_steps=8):
    for epoch in range(num_epochs):
        permutation = np.random.permutation(states.shape[0])
        states, actions = states[permutation], actions[permutation]

        total_d_loss, total_g_loss = 0, 0
        num_batches = 0

        for i in range(0, states.shape[0], batch_size * accumulation_steps):
            batch_d_loss, batch_g_loss = 0, 0
            
            for j in range(i, min(i + batch_size * accumulation_steps, states.shape[0]), batch_size):
                batch_states = torch.tensor(states[j:j+batch_size]).to(device)
                batch_actions = torch.tensor(actions[j:j+batch_size]).to(device)

                d_loss, g_loss = gan.train_step(batch_states, batch_actions)
                batch_d_loss += d_loss
                batch_g_loss += g_loss

            total_d_loss += batch_d_loss / accumulation_steps
            total_g_loss += batch_g_loss / accumulation_steps
            num_batches += 1

        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

def main():
    data = np.load("gameplay_data.npz")
    states, actions = data['states'], data['actions']
    states, actions = preprocess_data(states, actions)

    state_dim = states.shape[1]
    gan = SnakeGAN(state_dim)

    # Estimate memory usage
    # gen_memory = estimate_memory_usage(gan.generator, (gan.latent_dim + 4,), batch_size=4)
    # disc_memory = estimate_memory_usage(gan.discriminator, (state_dim + 4,), batch_size=4)
    # total_memory = gen_memory + disc_memory
    # print(f"Estimated VRAM usage: {total_memory:.2f} MB")

    train(gan, states, actions, batch_size=4, accumulation_steps=8)

    torch.save(gan.generator.state_dict(), "generator.pth")
    # files.download("generator.pth")

if __name__ == "__main__":
    main()