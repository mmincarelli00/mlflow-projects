"""
Synthetic per-UE RAN time series for testing the RLF-prediction pipeline.

NOT a substitute for real data (Colosseum / OpenRAN Gym / srsRAN / operator PM
feeds). It exists so the whole pipeline (features -> labels -> MLflow runs) can
be developed and debugged before real data is available. Column names are
illustrative; align them with the E2SM-KPM / 3GPP TS 28.552 measurements you
actually collect.

One row per (UE, second). Each UE lives in one serving cell. "Degradation
episodes" ramp the serving signal down while the neighbor improves; an episode
ends in an RLF (55%), a handover (25%) or a recovery (20%). A few RLFs occur
with no warning at all, so the task is not trivially solvable.
"""
import numpy as np
import pandas as pd

A3_OFFSETS = [2.0, 3.0, 4.0]      # dB
HYSTERESIS = [1.0, 2.0]           # dB
TTT_MS = [160, 320, 480]


def _ar1(n, phi, sigma, rng):
    x = np.zeros(n)
    e = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate_dataset(n_ues=60, n_seconds=3600, n_cells=12, seed=7,
                     episode_rate=1 / 400, sudden_rlf_rate=1 / 6000):
    rng = np.random.default_rng(seed)

    cell_a3 = rng.choice(A3_OFFSETS, n_cells)
    cell_hys = rng.choice(HYSTERESIS, n_cells)
    cell_ttt = rng.choice(TTT_MS, n_cells)
    cell_load = rng.uniform(0.25, 0.7, n_cells)

    t = np.arange(n_seconds)
    frames = []
    for ue in range(n_ues):
        n = n_seconds
        cell = int(rng.integers(n_cells))

        base_rsrp = -86 + _ar1(n, 0.995, 0.6, rng)
        base_nb = -96 + _ar1(n, 0.995, 0.6, rng)

        degr = np.zeros(n)
        rlf = np.zeros(n, dtype=int)
        ho = np.zeros(n, dtype=int)
        i = 0
        while i < n - 1:
            if rng.random() < episode_rate:
                end = min(i + int(rng.integers(10, 45)), n - 1)
                depth = rng.uniform(18, 32)
                degr[i:end + 1] = np.linspace(0, depth, end - i + 1)
                u = rng.random()
                if u < 0.55:
                    rlf[end] = 1
                elif u < 0.80:
                    ho[end] = 1
                i = end + 5
            else:
                i += 1
        sudden = (rng.random(n) < sudden_rlf_rate) & (degr == 0)
        rlf[sudden] = 1

        rsrp = base_rsrp - degr + rng.normal(0, 1.0, n)
        nb = base_nb + 0.35 * degr + rng.normal(0, 1.0, n)
        sinr = 18 + 0.8 * (rsrp + 86) + rng.normal(0, 1.2, n)
        rsrq = -10.5 + 0.12 * (rsrp + 86) + rng.normal(0, 0.4, n)
        ta = np.clip(np.round(20 + _ar1(n, 0.999, 0.3, rng) + 0.4 * degr
                              + rng.normal(0, 0.3, n)), 0, None)
        phr = 14 - 0.25 * degr + rng.normal(0, 1.0, n)
        ul_sinr = sinr - 3 + rng.normal(0, 1.5, n)
        cqi = np.clip(np.round(9 + (sinr - 15) / 2.5 + rng.normal(0, 0.7, n)), 1, 15)
        dl_bler = np.clip(0.04 + 0.012 * degr + rng.normal(0, 0.01, n), 0, 1)
        harq = np.clip(0.07 + 0.012 * degr + rng.normal(0, 0.015, n), 0, 1)

        gap = nb - rsrp
        lam = np.clip(0.01 + 0.03 * np.clip(gap - cell_a3[cell], 0, None), 0, 2)
        a3_events = rng.poisson(lam)

        load = np.clip(cell_load[cell]
                       + 0.15 * np.sin(2 * np.pi * (t + rng.integers(n)) / n)
                       + _ar1(n, 0.99, 0.01, rng), 0.02, 0.99)
        connected = rng.poisson(5 + 60 * load)

        frames.append(pd.DataFrame({
            "ue_id": ue, "cell_id": cell, "t": t,
            "rsrp": rsrp, "rsrq": rsrq, "sinr": sinr, "neighbor_rsrp": nb,
            "ta": ta, "phr": phr, "ul_sinr": ul_sinr, "cqi": cqi,
            "dl_bler": dl_bler, "harq_retx": harq,
            "a3_events": a3_events, "ho_cmd": ho,
            "prb_util": load, "connected_ues": connected,
            "a3_offset": cell_a3[cell], "hysteresis": cell_hys[cell],
            "ttt_ms": cell_ttt[cell],
            "hour_of_day": ((t + 8 * 3600) // 3600) % 24,
            "rlf": rlf,
        }))

    return (pd.concat(frames, ignore_index=True)
              .sort_values(["ue_id", "t"]).reset_index(drop=True))


if __name__ == "__main__":
    d = generate_dataset(n_ues=10, n_seconds=1800)
    print(d.shape, "RLF events:", int(d.rlf.sum()))
    print(d.head())
