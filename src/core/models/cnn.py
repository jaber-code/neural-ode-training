"""v_theta(s, a) -> ds/dt for pixel states -- a small conv encoder/decoder,
same role as VectorFieldMLP but for a flattened (H, W, C) frame instead of a
small physical-state vector. Registered as `model.name: cnn`.

Two things this handles differently from the MLP, both because the input is
an image and the action is a discrete index, not a continuous vector:
  - s is unflattened back to (B, C, H, W) using `frame_shape` (AtariPongDataset
    exposes this; run.py passes it through when a dataset has it) before
    going through conv layers, and the output is reflattened at the end so
    the (B, state_dim) contract every integrator/trainer expects is unchanged.
  - a is a discrete action id (see core/datasets/atari_pong.py's module
    docstring) cast to float by the dataset -- cast back to long here and
    looked up through an embedding table, not concatenated like a torque
    vector would be.

The encoder/decoder are architecturally symmetric (matching stride-2 conv /
stride-2 transposed-conv layer counts and channel counts), but 84 isn't a
power of 2, so the decoder's transposed-conv output size doesn't land back on
exactly 84x84 from stride/padding arithmetic alone -- a bilinear resize to
frame_shape at the very end (a no-op if it already matches) closes that gap
rather than hand-tuning padding to make the arithmetic exact.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODELS


@MODELS.register("cnn")
class VectorFieldCNN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        frame_shape: Tuple[int, int, int],
        num_actions: int = 6,     # Pong's minimal Gym/ALE action set -- confirm with np.unique(actions) on the real data
        embed_dim: int = 16,
        channels: Sequence[int] = (32, 64, 64),
    ):
        super().__init__()
        h, w, c = frame_shape
        assert h * w * c == state_dim, f"frame_shape {frame_shape} doesn't match state_dim {state_dim}"
        self.frame_shape = (h, w, c)

        self.action_embed = nn.Embedding(num_actions, embed_dim)

        enc_layers = []
        in_ch = c
        for out_ch in channels:
            enc_layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        with torch.no_grad():
            enc_out_shape = self.encoder(torch.zeros(1, c, h, w)).shape[1:]  # (C', H', W')
        self.enc_out_shape = enc_out_shape
        enc_channels = enc_out_shape[0]

        # action injected as extra broadcast channels (tiled across H'xW'), mixed with a
        # small conv -- NOT a flatten + Linear, which for a 64x11x11 map would need a
        # ~60M-parameter dense layer just to mix in a 16-dim embedding
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels + embed_dim, enc_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        dec_channels = list(channels[::-1][1:]) + [c]  # mirror encoder, ending back at c channels
        dec_layers = []
        in_ch = channels[-1]
        for out_ch in dec_channels:
            dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if out_ch != c:  # no ReLU on the final layer -- a pixel delta can be negative
                dec_layers.append(nn.ReLU())
            in_ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        b = s.shape[0]
        h, w, c = self.frame_shape
        img = s.view(b, h, w, c).permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        feat = self.encoder(img)  # (B, C', H', W')

        action_idx = a.reshape(b).long().clamp(0, self.action_embed.num_embeddings - 1)
        act_emb = self.action_embed(action_idx)  # (B, embed_dim)
        act_map = act_emb[:, :, None, None].expand(-1, -1, feat.shape[-2], feat.shape[-1])  # tile spatially

        mixed = self.bottleneck(torch.cat([feat, act_map], dim=1))

        delta = self.decoder(mixed)
        if delta.shape[-2:] != (h, w):
            delta = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)

        return delta.permute(0, 2, 3, 1).reshape(b, h * w * c)  # back to (B, state_dim)
