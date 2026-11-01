# Smart Underground Leakage & NRW Detector - Prototype ()

## Run (Windows / Mac / Linux)
```
pip install -r requirements.txt
python -c "import ml; ml.train()"     # optional: model train karke leak_model.joblib banata hai
python check.py                        # optional: pipeline self-test
streamlit run app.py
```
Browser me http://localhost:8501 khulega.

## Demo script (2 min)
1. Dashboard dikhao: sab zones green, flow/pressure normal.
2. Sidebar: Zone 3 select -> "Demand spike" : flow badhta hai, par alert NAHI aata (false alarm filter).
3. "Inject LEAK" : pressure girta hai, flow badhta hai -> ~5 sec me Zone 3 red, alert, loss counter chalu.
4. "Repair leak"  -> zone wapas green.
5. Bolo: real ESP32 sirf yahi (zone, flow, pressure) readings bhejega; baaki pipeline same rahegi.

## Files
simulator.py = fake sensors | ml.py = Random Forest + persistence filter | app.py = dashboard | check.py = test
