import torch
import torch.nn as nn


class Encoder_conv(nn.Module):
    def __init__(self):
        super().__init__()
        print("Building convolutional encoder...")
        print("Conv layers...")
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)

        print("FC layers...")
        self.fc1 = nn.Linear(4 * 4 * 32, 256, bias=False)
        self.fc2 = nn.Linear(256, 128, bias=False)

        self.relu = nn.ReLU()

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, nonlinearity="relu")

    def forward(self, x):
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))
        conv3_out_flat = torch.flatten(conv3_out, 1)
        fc1_out = self.relu(self.fc1(conv3_out_flat))
        fc2_out = self.relu(self.fc2(fc1_out))
        return fc2_out


class Encoder_mlp(nn.Module):
    def __init__(self):
        super(Encoder_mlp, self).__init__()
        print('Building MLP encoder...')
        # Fully-connected layers
        print('FC layers...')
        self.fc1 = nn.Linear(32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        # Nonlinearities
        self.relu = nn.ReLU()
        # Initialize parameters
        for name, param in self.named_parameters():
            # Initialize all biases to 0
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            # Initialize all pre-ReLU weights using Kaiming normal distribution
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    def forward(self, x):
        # Flatten image
        x_flat = torch.flatten(x, 1)
        # Fully-connected layers
        fc1_out = self.relu(self.fc1(x_flat))
        fc2_out = self.relu(self.fc2(fc1_out))
        fc3_out = self.relu(self.fc3(fc2_out))
        # Output
        z = fc3_out
        return z


class Encoder_rand(nn.Module):
    def __init__(self):
        super().__init__()
        print("Building random encoder...")
        self.fc1 = nn.Linear(32 * 32, 128, bias=False)
        self.relu = nn.ReLU()

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, nonlinearity="relu")

    def forward(self, x):
        x_flat = torch.flatten(x, 1)
        fc1_out = self.relu(self.fc1(x_flat)).detach()
        return fc1_out.detach()


class Model(nn.Module):
    def __init__(self, encoder_type):
        super().__init__()
        print("Building encoder...")
        if encoder_type == "conv":
            self.encoder = Encoder_conv()
        elif encoder_type == "mlp":
            self.encoder = Encoder_mlp()
        elif encoder_type == "rand":
            self.encoder = Encoder_rand()
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")

    def forward(self, x):
        return self.encoder(x)
