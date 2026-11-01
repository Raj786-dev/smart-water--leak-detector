"""ML: Random Forest (normal / leak / demand-spike) + persistence filter."""
import os
from collections import defaultdict, deque
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from simulator import Simulator, ZONES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "leak_model.joblib")


class FeatureBuilder:
    def __init__(self):
        self.h = defaultdict(lambda: deque(maxlen=3))

    def add(self, r):
        fd = r["flow"] / r["expected_flow"] - 1
        pd_ = r["pressure"] / r["expected_pressure"] - 1
        h = self.h[r["zone_id"]]
        prev_p = h[-1][1] if h else pd_
        h.append((fd, pd_))
        return [fd, pd_, np.mean([x[0] for x in h]), np.mean([x[1] for x in h]), pd_ - prev_p]


def train(verbose=True):
    X, y = [], []
    for ep in range(12):
        sim, fb, rng = Simulator(seed=ep, start_minute=int(ep * 97) % 1440), FeatureBuilder(), np.random.default_rng(100 + ep)
        leak_end = {}
        for t in range(1500):
            for z in ZONES:
                zid = z["id"]
                if zid in leak_end and t >= leak_end[zid]:
                    sim.clear_leak(zid); del leak_end[zid]
                if zid not in sim.leaks and rng.random() < 0.004:
                    sim.inject_leak(zid); leak_end[zid] = t + int(rng.integers(30, 90))
                if zid not in sim.spikes and rng.random() < 0.01:
                    sim.inject_spike(zid)
            for r in sim.step():
                X.append(fb.add(r)); y.append(r["label"])
    X, y = np.array(X), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    model = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=1, n_jobs=-1).fit(Xtr, ytr)
    if verbose:
        print(f"Rows: {len(y)} | normal={np.sum(y==0)} leak={np.sum(y==1)} spike={np.sum(y==2)}")
        print(classification_report(yte, model.predict(Xte), target_names=["normal", "leak", "demand_spike"], digits=3))
    joblib.dump(model, MODEL_PATH)
    return model


def load_or_train():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train(verbose=False)


class Detector:
    """Har reading pe leak probability nikalta hai; 3 lagataar anomalies = High Risk (false alerts kam)."""
    def __init__(self, model, threshold=0.6, persist=3):
        self.model, self.th, self.persist = model, threshold, persist
        self.fb, self.count, self.status = FeatureBuilder(), defaultdict(int), defaultdict(lambda: "Normal")

    def update(self, readings):
        X = np.array([self.fb.add(r) for r in readings])
        probs = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        li = classes.index(1)
        out = []
        for r, p in zip(readings, probs):
            zid, pl = r["zone_id"], float(p[li])
            self.count[zid] = min(6, self.count[zid] + 1) if pl >= self.th else max(0, self.count[zid] - 1)
            c = self.count[zid]
            new = "High Risk" if c >= self.persist else ("Watch" if c >= 1 else "Normal")
            out.append({**r, "leak_prob": pl, "status": new, "new_alert": new == "High Risk" and self.status[zid] != "High Risk"})
            self.status[zid] = new
        return out

    def reset(self):
        self.__init__(self.model, self.th, self.persist)
