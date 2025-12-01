import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bayesian_layers import BayesianLinear


class BCNN_LSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 64,
        n_timepoints: int = 78,
        spatial_kernels: int = 10,
        temporal_kernels: int = 50,
        temporal_kernel_width: int = 13,
        temporal_stride: int = 13,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        bayes_fc_hidden: int = 100,
        prior_sigma1: float = 0.1,
        prior_sigma2: float = 0.0005,
        prior_pi: float = 0.5,
        init_std: float = 0.05,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ---------- Spatial convolution ----------
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=spatial_kernels,
            kernel_size=(n_channels, 1),
        )

        # ---------- Temporal convolution ----------
        self.temporal_conv = nn.Conv2d(
            in_channels=spatial_kernels,
            out_channels=temporal_kernels,
            kernel_size=(1, temporal_kernel_width),
            stride=(1, temporal_stride),
        )
        self.temporal_bn = nn.BatchNorm2d(temporal_kernels)
        self.temporal_relu = nn.ReLU()

        # output temporal dimension
        temp_out = (n_timepoints - temporal_kernel_width) // temporal_stride + 1

        # ---------- LSTM ----------
        self.lstm = nn.LSTM(
            input_size=temporal_kernels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---------- Bayesian fully connected ----------
        self.fc1 = BayesianLinear(
            lstm_out_dim,
            bayes_fc_hidden,
            prior_sigma1,
            prior_sigma2,
            prior_pi,
            init_std,
        )
        self.fc2 = BayesianLinear(
            bayes_fc_hidden,
            2,
            prior_sigma1,
            prior_sigma2,
            prior_pi,
            init_std,
        )

    def forward(self, x):
        # x: (batch, channels, time)
        # x = x.unsqueeze(1)                # -> (batch, 1, channels, time)
        x = self.spatial_conv(x)          # -> (batch, spatial_kernels, 1, time)
        x = self.temporal_conv(x)         # -> (batch, temporal_kernels, 1, new_time)
        x = self.temporal_bn(x)
        x = self.temporal_relu(x)
        x = x.squeeze(2)                  # -> (batch, temporal_kernels, new_time)
        x = x.permute(0, 2, 1)            # -> (batch, new_time, temporal_kernels)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]            # take last timestep output
        x = self.dropout(x)

        # Bayesian fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

    @torch.no_grad()
    def predict_proba_mc(self, x, n_samples=20, device="cuda"):
        """
        Monte Carlo prediction: run n_samples stochastic forward passes
        and return averaged softmax probabilities.
        """
        self.eval()
        x = x.to(device)
        logits_mc = []
        for _ in range(n_samples):
            logits = self(x)
            logits_mc.append(F.softmax(logits, dim=1))
        probs = torch.mean(torch.stack(logits_mc), dim=0)
        return probs

    
    
