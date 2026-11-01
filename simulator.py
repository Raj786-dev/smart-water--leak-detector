"""Sensor simulator: yeh ESP32 flow + pressure sensors ki jagah data banata hai."""
import numpy as np

# Zones (coordinates approximate, Delhi ke aas-paas)
ZONES = [
    {"id": 1, "name": "Zone 1 - Yamuna Vihar", "lat": 28.7040, "lon": 77.2480, "base_flow": 120, "base_pressure": 3.6},
    {"id": 2, "name": "Zone 2 - Bhajanpura",   "lat": 28.7010, "lon": 77.2610, "base_flow": 95,  "base_pressure": 3.4},
    {"id": 3, "name": "Zone 3 - Gokalpuri",    "lat": 28.7050, "lon": 77.2760, "base_flow": 140, "base_pressure": 3.5},
    {"id": 4, "name": "Zone 4 - Maujpur",      "lat": 28.6940, "lon": 77.2770, "base_flow": 110, "base_pressure": 3.3},
    {"id": 5, "name": "Zone 5 - Jafrabad",     "lat": 28.6880, "lon": 77.2650, "base_flow": 130, "base_pressure": 3.7},
    {"id": 6, "name": "Zone 6 - Seelampur",    "lat": 28.6700, "lon": 77.2670, "base_flow": 100, "base_pressure": 3.4},
]
LINKS = [(1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]   # pipe connections (map pe lines)
ZONE_BY_ID = {z["id"]: z for z in ZONES}


def fmt_time(minute):
    m = int(minute) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def expected(zone, minute):
    """Time-of-day ke hisaab se normal flow/pressure (subah 7 aur shaam 7 peak)."""
    h = (minute % 1440) / 60
    d = 1 + 0.25 * np.cos(2 * np.pi * (h - 7) / 12)
    return zone["base_flow"] * d, zone["base_pressure"] * (1 - 0.2 * (d - 1))


class Simulator:
    def __init__(self, seed=None, start_minute=120):   # 02:00 AM se start (night leak use case)
        self.rng = np.random.default_rng(seed)
        self.start = start_minute
        self.reset()

    def reset(self):
        self.minute = self.start
        self.leaks = {}    # zone_id -> {"ticks", "f", "p"}
        self.spikes = {}   # zone_id -> {"left", "f", "p"}

    def inject_leak(self, zid):
        self.leaks[zid] = {"ticks": 0, "f": self.rng.uniform(0.12, 0.25), "p": self.rng.uniform(0.12, 0.22)}

    def clear_leak(self, zid):
        self.leaks.pop(zid, None)

    def inject_spike(self, zid):
        self.spikes[zid] = {"left": int(self.rng.integers(3, 7)), "f": self.rng.uniform(0.2, 0.35), "p": self.rng.uniform(0.02, 0.06)}

    def step(self):
        out = []
        for z in ZONES:
            ef, ep = expected(z, self.minute)
            flow, pres, label = ef, ep, 0
            if z["id"] in self.leaks:
                lk = self.leaks[z["id"]]
                sev = min(1.0, lk["ticks"] / 5)          # leak dheere dheere badhta hai
                flow *= 1 + lk["f"] * sev
                pres *= 1 - lk["p"] * sev
                label = 1 if sev >= 0.4 else 0
                lk["ticks"] += 1
            if z["id"] in self.spikes:
                sp = self.spikes[z["id"]]
                flow *= 1 + sp["f"]
                pres *= 1 - sp["p"]
                label = 2
                sp["left"] -= 1
                if sp["left"] <= 0:
                    del self.spikes[z["id"]]
            flow *= self.rng.normal(1, 0.02)
            pres *= self.rng.normal(1, 0.012)
            out.append({"zone_id": z["id"], "minute": self.minute, "flow": flow, "pressure": pres,
                        "expected_flow": ef, "expected_pressure": ep, "label": label})
        self.minute += 1
        return out
