"""Quick self-test: dashboard ke bina pipeline check karta hai."""
from simulator import Simulator
from ml import load_or_train, Detector

m = load_or_train(); sim = Simulator(seed=7); det = Detector(m)
for _ in range(40): res = det.update(sim.step())
print("Normal 40 ticks: false High Risk =", sum(r["status"] == "High Risk" for r in res))
sim.inject_spike(2)
fa = 0
for _ in range(8): fa += sum(r["status"] == "High Risk" for r in det.update(sim.step()))
print("Demand spike Zone 2: High Risk alerts =", fa)
for _ in range(10): det.update(sim.step())
sim.inject_leak(3)
for t in range(1, 25):
    res = det.update(sim.step())
    z3 = [r for r in res if r["zone_id"] == 3][0]
    if z3["status"] == "High Risk":
        print(f"Leak Zone 3 detected after {t} ticks (prob={z3['leak_prob']:.2f})"); break
else: print("NOT detected")
print("Other zones flagged:", [r["zone_id"] for r in res if r["status"] == "High Risk" and r["zone_id"] != 3])
