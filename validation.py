"""Validation: detector ki asli performance naapta hai (simulated data pe).
Chalao:  python validation.py      -> reports/ folder me report + chart banta hai
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator import Simulator, ZONES
from ml import load_or_train, Detector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(OUT, exist_ok=True)
model = load_or_train()
try:
    model.n_jobs = 1          # chhote batches pe tez
except Exception:
    pass


def noisy(readings, rng, extra):
    if extra:
        for r in readings:
            r["flow"] *= rng.normal(1, extra); r["pressure"] *= rng.normal(1, extra / 2)
    return readings


def false_alarms(extra_noise, episodes=8, ticks=300):
    """Koi leak nahi: sirf normal + random demand spikes. Kitne (galat) High Risk alerts aaye?"""
    alerts = 0
    for ep in range(episodes):
        sim, det, rng = Simulator(seed=500 + ep, start_minute=ep * 170), Detector(model), np.random.default_rng(900 + ep)
        for _ in range(ticks):
            for z in ZONES:
                if z["id"] not in sim.spikes and rng.random() < 0.01:
                    sim.inject_spike(z["id"])
            alerts += sum(r["new_alert"] for r in det.update(noisy(sim.step(), rng, extra_noise)))
    return alerts, episodes * ticks * len(ZONES)


def detection(size, trials=15, warmup=40, horizon=40):
    """Leak size (flow badhna) ke hisaab se: kitni baar pakda, aur kitne ticks me."""
    found, delays, wrong = 0, [], 0
    for t in range(trials):
        rng = np.random.default_rng(7000 + t)
        sim, det = Simulator(seed=3000 + t, start_minute=int(rng.integers(0, 1400))), Detector(model)
        zid = int(rng.integers(1, len(ZONES) + 1))
        for _ in range(warmup):
            det.update(sim.step())
        sim.inject_leak(zid); sim.leaks[zid]["f"], sim.leaks[zid]["p"] = size, size
        hit = None
        for k in range(1, horizon + 1):
            for r in det.update(sim.step()):
                if r["status"] == "High Risk":
                    if r["zone_id"] == zid and hit is None: hit = k
                    elif r["zone_id"] != zid: wrong += 1
        if hit: found += 1; delays.append(hit)
    return found / trials, (float(np.mean(delays)) if delays else None), wrong


t0 = time.time()
res = {"false_alarms": [], "detection": []}
for n in (0.0, 0.01, 0.02):
    a, zt = false_alarms(n)
    res["false_alarms"].append({"extra_noise": n, "alerts": a, "zone_ticks": zt, "per_1000_zone_ticks": round(a / zt * 1000, 3)})
    print(f"noise +{n:.2f}: false alerts = {a} in {zt} zone-ticks", flush=True)
for s in (0.03, 0.05, 0.08, 0.12, 0.20):
    rate, delay, wrong = detection(s)
    res["detection"].append({"leak_size": s, "detection_rate": rate, "mean_delay_ticks": delay, "wrong_zone_alerts": wrong})
    print(f"leak {s*100:>4.0f}%: detected {rate*100:.0f}% | mean delay {delay if delay is None else round(delay,1)} ticks | wrong-zone {wrong}", flush=True)

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)

# chart
d = res["detection"]; xs = [f"{x['leak_size']*100:.0f}%" for x in d]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(xs, [x["detection_rate"] * 100 for x in d], color="#2a7fbf")
ax.set_ylim(0, 105); ax.set_xlabel("Leak size (extra flow, with matching pressure drop)"); ax.set_ylabel("Detection rate (%)")
ax2 = ax.twinx(); ax2.plot(xs, [x["mean_delay_ticks"] if x["mean_delay_ticks"] else np.nan for x in d], "o-", color="#d62728")
ax2.set_ylabel("Mean detection delay (ticks = sim. minutes)")
ax.set_title("Leak detection vs leak size (simulated data)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "detection_by_leak_size.png"), dpi=150)

fa = res["false_alarms"]
md = ["# Validation report (simulated data)", "", "> Sab numbers **simulated scenarios** par hain, jinpe model train hua hai. Real pipeline me performance alag hogi; pilot me real data se dobara naapna hoga.", "",
      "## False alarms (koi leak nahi; normal + random demand spikes)", "", "| Extra sensor noise | False alerts | Zone-ticks | Per 1000 zone-ticks |", "|---|---|---|---|"]
md += [f"| +{f['extra_noise']*100:.0f}% | {f['alerts']} | {f['zone_ticks']} | {f['per_1000_zone_ticks']} |" for f in fa]
md += ["", "## Leak detection by leak size", "", "| Leak size | Detection rate | Mean delay (ticks) | Wrong-zone alerts |", "|---|---|---|---|"]
md += [f"| {x['leak_size']*100:.0f}% | {x['detection_rate']*100:.0f}% | {'-' if x['mean_delay_ticks'] is None else round(x['mean_delay_ticks'], 1)} | {x['wrong_zone_alerts']} |" for x in d]
md += ["", "![chart](detection_by_leak_size.png)", "", "1 tick = 1 simulated minute. Alert tab aata hai jab 3 lagataar readings anomalous hon (persistence filter)."]
open(os.path.join(OUT, "validation_report.md"), "w", encoding="utf-8").write("\n".join(md))
print(f"done in {time.time() - t0:.0f}s -> reports/")
