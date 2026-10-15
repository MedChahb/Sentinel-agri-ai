"""Part 2: the GRU, a sequence model that fills NDVI gaps over a whole season.

Kept apart from `gapfill` so that the gap-filling core, used by the API, does not need PyTorch.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from sentinel_agri.gapfill import STEP_DAYS, Grid, make_mask, neighbours


class GapFillGRU(nn.Module):
    """Bidirectional GRU over the whole season.

    It sees the radar and the visible NDVI before and after each step, and predicts NDVI everywhere.
    """

    def __init__(self, n_inputs: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            n_inputs,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


def sequence_inputs(
    grid: Grid,
    hidden: np.ndarray,
    s1_mean,
    s1_std,
    use_radar: bool = True,
    use_date: bool = True,
) -> torch.Tensor:
    """(P, T, 9) inputs: standardised radar (3), day of the year (2), visible NDVI and its flag (2),
    days since the previous visible observation (1), crop (1). Hidden values are set to 0.

    With `use_radar=False`, the radar channels are set to 0: the same network without radar.
    With `use_date=False`, the day-of-year channels are set to 0.
    """
    visible = grid.observed & ~hidden
    prev, _ = neighbours(visible)
    since = np.where(prev >= 0, np.arange(grid.ndvi.shape[1])[None, :] - prev, 10) * STEP_DAYS / 50
    doy = grid.doy
    n_p, n_t = grid.ndvi.shape
    x = np.concatenate(
        [
            np.nan_to_num((grid.s1 - s1_mean) / s1_std) * use_radar,  # no radar: average everywhere
            np.broadcast_to(np.sin(2 * np.pi * doy / 365) * use_date, (n_p, n_t))[..., None],
            np.broadcast_to(np.cos(2 * np.pi * doy / 365) * use_date, (n_p, n_t))[..., None],
            np.where(visible, np.nan_to_num(grid.ndvi), 0)[..., None],
            visible[..., None].astype(float),
            since[..., None],
            np.broadcast_to((grid.crops == "Winter rapeseed")[:, None], (n_p, n_t))[
                ..., None
            ].astype(float),
        ],
        axis=-1,
    )
    return torch.tensor(x, dtype=torch.float32)


def train_gru(
    grid: Grid,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    val_hidden: np.ndarray,
    epochs: int = 300,
    lr: float = 3e-3,
    seed: int = 0,
    log_every: int = 25,
    use_radar: bool = True,
    use_date: bool = True,
):
    """Train on the training plots and keep the epoch with the lowest validation MAE.

    The training plots get new random gaps at every epoch, so the model cannot memorise a mask; the
    validation plots keep fixed gaps.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    s1_mean = np.nanmean(grid.s1[train_rows], axis=(0, 1))
    s1_std = np.nanstd(grid.s1[train_rows], axis=(0, 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"  # a GPU makes 1,000+ plots practical
    model = GapFillGRU(n_inputs=9).to(device)
    model.use_radar, model.use_date = use_radar, use_date
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    target = torch.tensor(np.nan_to_num(grid.ndvi), dtype=torch.float32, device=device)
    val_x = sequence_inputs(grid, val_hidden, s1_mean, s1_std, use_radar, use_date)
    val_x = val_x[val_rows].to(device)
    val_mask = torch.tensor(val_hidden[val_rows], device=device)
    best, best_state, history = np.inf, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        hidden = np.zeros_like(grid.observed)
        hidden[train_rows] = np.stack([make_mask(grid.observed[p], rng) for p in train_rows])
        x = sequence_inputs(grid, hidden, s1_mean, s1_std, use_radar, use_date)
        x = x[train_rows].to(device)
        mask = torch.tensor(hidden[train_rows], device=device)
        loss = (model(x) - target[train_rows]).abs()[mask].mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_mae = float((model(val_x) - target[val_rows]).abs()[val_mask].mean())
        history.append({"epoch": epoch, "train_mae": loss.item(), "val_mae": val_mae})
        if val_mae < best:
            best, best_state = val_mae, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % log_every == 0:
            print(f"epoch {epoch:>3} | train MAE {loss.item():.4f} | validation MAE {val_mae:.4f}")
    model.load_state_dict(best_state)
    model.cpu()
    return model, (s1_mean, s1_std), pd.DataFrame(history)


def predict_gru(model, grid: Grid, hidden: np.ndarray, scaling) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = sequence_inputs(grid, hidden, *scaling, model.use_radar, model.use_date)
        return model(inputs).numpy()
