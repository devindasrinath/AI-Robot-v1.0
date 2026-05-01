"""
ManualGRUCell behavior policy for Nexu.

Architecture avoids nn.GRU (not supported by tpu-mlir).
Uses explicit matrix multiplications that map cleanly to tpu-mlir Matmul ops.

Inputs  (both required at every step):
  obs       [batch, 1, OBS_SIZE]    — current observation (seq_len=1)
  hidden_in [batch, HIDDEN_SIZE]    — GRU hidden state from previous step

Outputs:
  action_logits [batch, NUM_ACTIONS]
  hidden_out    [batch, HIDDEN_SIZE]
"""

import torch
import torch.nn as nn
import math

OBS_SIZE    = 16
HIDDEN_SIZE = 64
NUM_ACTIONS = 12


class ManualGRUCell(nn.Module):
    """
    GRU cell as explicit linear ops — safe for tpu-mlir.

    Gates: r = sigmoid(Wr*x + Ur*h + br)
           z = sigmoid(Wz*x + Uz*h + bz)
           n = tanh(Wn*x + r*(Un*h + bn))
       h' = (1-z)*n + z*h
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        I = input_size
        # Reset gate
        self.Wr = nn.Linear(I, H, bias=True)
        self.Ur = nn.Linear(H, H, bias=False)
        # Update gate
        self.Wz = nn.Linear(I, H, bias=True)
        self.Uz = nn.Linear(H, H, bias=False)
        # New gate
        self.Wn = nn.Linear(I, H, bias=True)
        self.Un = nn.Linear(H, H, bias=True)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """x: [batch, input_size]   h: [batch, hidden_size]"""
        r = torch.sigmoid(self.Wr(x) + self.Ur(h))
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))
        n = torch.tanh(self.Wn(x) + r * (self.Un(h)))
        h_new = (1.0 - z) * n + z * h
        return h_new


class NexuGRUPolicy(nn.Module):
    """
    Nexu behavior GRU policy.

    ONNX export shape:
      obs       [1, 1, OBS_SIZE]   → squeeze seq dim → [1, OBS_SIZE]
      hidden_in [1, HIDDEN_SIZE]
    Output:
      action_logits [1, NUM_ACTIONS]
      hidden_out    [1, HIDDEN_SIZE]
    """

    def __init__(self):
        super().__init__()
        self.gru  = ManualGRUCell(OBS_SIZE, HIDDEN_SIZE)
        self.head = nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)

    def forward(self, obs: torch.Tensor, hidden_in: torch.Tensor):
        """
        obs:       [batch, 1, OBS_SIZE]
        hidden_in: [batch, HIDDEN_SIZE]
        """
        x = obs[:, 0, :]                          # [batch, OBS_SIZE]
        h = self.gru(x, hidden_in)                # [batch, HIDDEN_SIZE]
        logits = self.head(h)                      # [batch, NUM_ACTIONS]
        return logits, h


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = NexuGRUPolicy()
    model.eval()

    obs    = torch.randn(1, 1, OBS_SIZE)
    hidden = torch.zeros(1, HIDDEN_SIZE)

    logits, h_out = model(obs, hidden)
    print(f"obs:          {obs.shape}")
    print(f"hidden_in:    {hidden.shape}")
    print(f"action_logits:{logits.shape}")
    print(f"hidden_out:   {h_out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    assert logits.shape == (1, NUM_ACTIONS)
    assert h_out.shape  == (1, HIDDEN_SIZE)
    print("Architecture OK")
