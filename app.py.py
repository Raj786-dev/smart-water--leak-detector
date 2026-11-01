"""Smart Underground Leakage & NRW Detector - Streamlit dashboard."""
import json, time, urllib.request
from collections import deque
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from simulator import Simulator, ZONES, LINKS, ZONE_BY_ID, fmt_time, expected
from ml import load_or_train, Detector

st.set_page_config(page_title="Smart Leak & NRW Detector", page_icon="💧", layout="wide")
API_URL = "http://127.0.0.1:8000"
SECS_PER_TICK = 60   # 1 tick = 1 simulated minute
DEMO_SCRIPT = {   # tick -> (action, zone, message)  | 1 tick ~ 1 refresh
    3:  ("spike", 3, "Demand spike in Zone 3: alert raised (not a leak)"),
    20: ("leak", 1, "Leak injected in Zone 1: turns yellow, then red + alert"),
    52: ("repair", 1, "Repair: Zone 1 back to green"),
}
DEMO_LEN = 70
COLORS = {"Normal": "#2ca02c", "Watch": "#f2b01e", "High Risk": "#d62728"}
ICON = {"Normal": "🟢", "Watch": "🟡", "High Risk": "🔴"}


@st.cache_resource
def get_model():
    return load_or_train()


def reset_all():
    ss = st.session_state
    ss.sim, ss.det = Simulator(seed=None), Detector(get_model())
    ss.hist = {z["id"]: deque(maxlen=120) for z in ZONES}
    ss.alerts, ss.last, ss.loss_l, ss.last_seq, ss.prev_status, ss.demo = [], [], 0.0, -1, {}, None


if "sim" not in st.session_state:
    reset_all()
ss = st.session_state

# ---------------- Sidebar: demo controls ----------------
with st.sidebar:
    st.header("🎛️ Demo Controls")
    st.radio("Data source", ["Simulator (demo)", "Live API (ESP32 connected)"], key="source", on_change=reset_all)
    live = ss.source.startswith("Live")
    st.toggle("▶️ Live simulation", value=True, key="running")
    refresh = st.slider("Refresh interval (seconds)", 0.5, 3.0, 1.0, 0.5)
    zid = st.selectbox("Zone", [z["id"] for z in ZONES], format_func=lambda i: ZONE_BY_ID[i]["name"], key="zone_sel")
    if live:
        st.info("Live mode: data is coming from the API (port 8000). Trigger leak/spike/repair from the `send_sensor.py` terminal: `l 3`, `s 3`, `r 3`.")
    else:
        if st.button("💥 Inject LEAK", use_container_width=True, type="primary"):
            ss.sim.inject_leak(zid)
        if st.button("⚡ Demand spike (NOT a leak)", use_container_width=True):
            ss.sim.inject_spike(zid)
        if st.button("🔧 Repair leak", use_container_width=True):
            ss.sim.clear_leak(zid)
        if st.button("🎬 Auto demo (~70 sec)", use_container_width=True):
            reset_all(); ss.demo = {"t": 0, "msg": "Normal state: all zones green"}
    if st.button("♻️ Reset demo", use_container_width=True):
        reset_all(); st.rerun()
    st.download_button("⬇️ Alerts CSV", data=pd.DataFrame(ss.alerts).to_csv(index=False), file_name="leak_alerts.csv",
                       mime="text/csv", use_container_width=True, disabled=not ss.alerts)
    st.radio("Map view", ["Schematic map (offline)", "Street map (online)"], key="mapmode")
    price = st.number_input("Water cost (₹/KL, estimated)", 1.0, 100.0, 20.0)
    st.caption("Sensor data is simulated. A real ESP32 will send the same readings (flow, pressure).")


def fetch_state():
    """Live mode: reads the detection result from the API (the ESP32 buzzer reads the same one)."""
    try:
        with urllib.request.urlopen(API_URL + "/state", timeout=1) as f:
            d = json.loads(f.read())
    except Exception:
        return None
    if d["version"] == ss.last_seq or not d["results"]:
        return None
    ss.last_seq = d["version"]
    prev, out = ss.get("prev_status", {}), []
    for r in d["results"]:
        r["tlabel"] = time.strftime("%H:%M:%S")
        r["new_alert"] = r["status"] == "High Risk" and prev.get(r["zone_id"]) != "High Risk"
        out.append(r)
    ss.prev_status = {r["zone_id"]: r["status"] for r in out}
    return out


def demo_tick():
    d = ss.get("demo")
    if not d:
        return
    d["t"] += 1
    ev = DEMO_SCRIPT.get(d["t"])
    if ev:
        act, z, msg = ev
        {"leak": ss.sim.inject_leak, "spike": ss.sim.inject_spike, "repair": ss.sim.clear_leak}[act](z)
        d["msg"] = msg
    if d["t"] >= DEMO_LEN:
        ss.demo = None


def step():
    if ss.get("source", "").startswith("Live"):
        results = fetch_state()
        if results is None:
            return
    else:
        demo_tick()
        readings = ss.sim.step()
        for r in readings:
            r["tlabel"] = fmt_time(r["minute"])
        results = ss.det.update(readings)
    ss.last = results
    for r in results:
        ss.hist[r["zone_id"]].append(r)
        excess = max(r["flow"] - r["expected_flow"], 0)
        if r["status"] == "High Risk":
            ss.loss_l += excess * SECS_PER_TICK
        if r["new_alert"]:
            ss.alerts.insert(0, {"Time": fmt_time(r["minute"]), "Zone": ZONE_BY_ID[r["zone_id"]]["name"],
                                 "Confidence": f"{r['leak_prob']*100:.0f}%", "Extra flow (L/s)": round(excess, 1),
                                 "Pressure drop": f"{(1 - r['pressure']/r['expected_pressure'])*100:.0f}%"})


def vlat(lat):
    return 28.685 + (lat - 28.6875) * 0.65   # map display only


def label_positions():
    """Places each zone name where the pipe lines are farthest away (map display only)."""
    import math
    names = ["middle right", "top right", "top center", "top left", "middle left", "bottom left", "bottom center", "bottom right"]
    lon_span = max(z["lon"] for z in ZONES) - min(z["lon"] for z in ZONES)
    lat_span = max(vlat(z["lat"]) for z in ZONES) - min(vlat(z["lat"]) for z in ZONES)
    out = []
    for z in ZONES:
        angs = []
        for a, b in LINKS:
            if z["id"] in (a, b):
                o = ZONE_BY_ID[b if a == z["id"] else a]
                dx, dy = (o["lon"] - z["lon"]) / lon_span * 700, (vlat(o["lat"]) - vlat(z["lat"])) / lat_span * 460
                angs.append(math.degrees(math.atan2(dy, dx)) % 360)
        def clearance(k):
            return min([min(abs(k * 45 - t) % 360, 360 - abs(k * 45 - t) % 360) for t in angs] or [180])
        out.append(names[max(range(8), key=clearance)])
    return out


def line_fig(zone_id, key, exp_key, title, unit, color):
    h = list(ss.hist[zone_id])
    fig = go.Figure()
    fig.add_scatter(x=[r["tlabel"] for r in h], y=[r[key] for r in h], name="Actual", line=dict(color=color, width=2))
    fig.add_scatter(x=[r["tlabel"] for r in h], y=[r[exp_key] for r in h], name="Expected (normal)",
                    line=dict(color="gray", dash="dash"))
    fig.update_layout(title=title, yaxis_title=unit, height=260, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def live_view():
    if ss.running:
        step()
    res = ss.last
    st.title("💧 Smart Underground Leakage & NRW Detector")
    if not res:
        st.info("Waiting for sensor data... run `python send_sensor.py`." if ss.source.startswith("Live") else "Simulation is starting..."); return
    st.caption(f"Simulated time: **{fmt_time(res[0]['minute'])}**")

    if ss.get("demo"):
        st.info(f"🎬 Auto demo ({ss.demo['t']}/{DEMO_LEN}): {ss.demo['msg']}")
    if any(r.get("calibrating") for r in res):
        st.info("📟 ESP32 zone is calibrating (~15 sec). Keep the sensors in the NORMAL state for now.")
    risk = [r for r in res if r["status"] == "High Risk"]
    loss_now = sum(max(r["flow"] - r["expected_flow"], 0) for r in risk)
    tot_flow = sum(r["flow"] for r in res)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active leak zones", len(risk))
    c2.metric("Leak loss now", f"{loss_now:.1f} L/s")
    c3.metric("Network leak loss", f"{loss_now / tot_flow * 100:.1f}%")
    c4.metric("Water lost so far", f"{ss.loss_l / 1000:.1f} KL")
    c5.metric("Estimated financial loss", f"₹{ss.loss_l / 1000 * price:,.0f}")

    left, right = st.columns([3, 2])
    with left:
        street = ss.get("mapmode", "").startswith("Street")
        fig = go.Figure()
        Sc = go.Scattermap if street else go.Scatter
        xs = lambda z: z["lon"]
        ys = lambda z: vlat(z["lat"])
        def pts(zs):
            return ({"lat": [ys(z) for z in zs], "lon": [xs(z) for z in zs]} if street
                    else {"x": [xs(z) for z in zs], "y": [ys(z) for z in zs]})
        for a, b in LINKS:
            fig.add_trace(Sc(**pts([ZONE_BY_ID[a], ZONE_BY_ID[b]]), mode="lines",
                             line=dict(width=4, color="#4a90d9"), hoverinfo="skip", showlegend=False))
        zs = [ZONE_BY_ID[r["zone_id"]] for r in res]
        fig.add_trace(Sc(**pts(zs), mode="markers+text", text=[f"Z{r['zone_id']}" for r in res],
                         textfont=dict(size=13, color="white"),
                         marker=dict(size=[48 if r["status"] == "High Risk" else 38 for r in res],
                                     color=[COLORS[r["status"]] for r in res]),
                         hovertext=[f"{ZONE_BY_ID[r['zone_id']]['name']}<br>{r['status']} ({r['leak_prob']*100:.0f}%)" for r in res],
                         hoverinfo="text", showlegend=False))
        lats, lons = [ys(z) for z in ZONES], [xs(z) for z in ZONES]
        if street:
            fig.update_layout(map=dict(style="open-street-map",
                                       bounds=dict(west=min(lons) - 0.03, east=max(lons) + 0.03,
                                                   south=min(lats) - 0.012, north=max(lats) + 0.012)),
                              uirevision="keep", height=460, margin=dict(l=0, r=0, t=0, b=0))
        else:
            fig.add_trace(go.Scatter(x=[xs(z) for z in ZONES], y=[ys(z) for z in ZONES], mode="markers+text",
                                     text=[z["name"].split(" - ")[1] for z in ZONES],
                                     textposition=label_positions(), textfont=dict(size=11, color="#9aa4b2"),
                                     marker=dict(size=38, opacity=0), hoverinfo="skip", showlegend=False))
            fig.update_layout(xaxis=dict(visible=False, range=[min(lons) - 0.008, max(lons) + 0.008]),
                              yaxis=dict(visible=False, range=[min(lats) - 0.006, max(lats) + 0.005]),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              height=460, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, key="map")
    with right:
        st.subheader("Zone status")
        st.dataframe(pd.DataFrame([{"": ICON[r["status"]], "Zone": ZONE_BY_ID[r["zone_id"]]["name"] + (" 📟" if r.get("hw") else ""), "Status": r["status"],
                                    "Flow (L/s)": round(r["flow"], 1), "Pressure (bar)": round(r["pressure"], 2),
                                    "Leak probability": f"{r['leak_prob']*100:.0f}%"} for r in res]),
                     hide_index=True, use_container_width=True, height=250)
        st.subheader("🚨 Alerts")
        if ss.alerts:
            st.dataframe(pd.DataFrame(ss.alerts[:6]), hide_index=True, use_container_width=True)
        else:
            st.caption("No active alerts.")

    sel = st.session_state.get("zone_sel", 1)
    st.subheader(f"📈 Live sensor readings - {ZONE_BY_ID[sel]['name']}")
    a, b = st.columns(2)
    a.plotly_chart(line_fig(sel, "flow", "expected_flow", "Flow", "L/s", "#1f77b4"), use_container_width=True, key="flow")
    b.plotly_chart(line_fig(sel, "pressure", "expected_pressure", "Pressure", "bar", "#ff7f0e"), use_container_width=True, key="pres")


st.fragment(run_every=refresh if ss.running else None)(live_view)()
