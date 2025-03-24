import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.rnn = nn.LSTM(512 * 8, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x = x.view(batch, width, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
