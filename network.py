from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder_hidden = nn.Linear(in_features=kwargs["input_shape"], out_features=128)
        self.encoder_output = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden = nn.Linear(in_features=128, out_features=128)
        self.decoder_output = nn.Linear(in_features=128, out_features=kwargs["input_shape"])
        self.linear_relu_stack = nn.Sequential(
            self.encoder_hidden,
            nn.ReLU(),
            self.encoder_output,
            nn.ReLU(),
            self.decoder_hidden,
            nn.ReLU(),
            self.decoder_output,
            nn.ReLU()
        )

    def forward(self, features):
        features = self.flatten(features)
        return self.linear_relu_stack(features)
