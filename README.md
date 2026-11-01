Smart Underground Leakage & NRW Detector

Low-cost IoT sensors and machine learning to detect hidden underground water leaks by zone and estimate Non-Revenue Water (NRW).

Team Dora Ducks: Raj Verma (Team Leader), Saurabh Tiwari, Vinish Kumar Kashyap, Shorya Mahajan

Category: Hybrid (Hardware + Software)

Overview

Leaks in underground pipes often stay unseen until they surface or supply fails. Utilities usually cannot tell where a leak is or what it costs, so crews end up searching the whole network.

This prototype monitors flow and pressure for each District Metered Area (DMA) zone, learns what normal behaviour looks like, and flags abnormal patterns as a leak. A live dashboard shows the suspect zone, raises alerts, and estimates the water and money lost.

How It Works
Sense: flow and pressure sensors per zone, read by an ESP32.
Analyse: a Random Forest model classifies each reading against expected hydraulic behaviour.
Filter: a persistence filter separates real leaks from short demand spikes, which reduces false alarms.
Localise: the dashboard map turns the affected zone red and fires an alert.
Quantify: excess flow is accumulated into estimated water lost (KL) and cost (INR).

A leak shows up as pressure falling while flow rises. A demand spike raises flow without the same pressure drop, so it is not reported as a leak.

Features
Live dashboard with zone map (offline schematic or street map), zone status table, and flow/pressure charts against expected values
Leak injection, demand spike (not a leak), and repair controls
One-click auto demo (about 70 seconds)
Real-time metrics: active leak zones, leak loss rate, network leak loss, water lost, estimated money lost
Alert log with confidence, extra flow, and pressure drop, with CSV export
Two data sources: built-in simulator, or live readings from an ESP32 through an API
Project Structure
File	Purpose
app.py	Streamlit dashboard
simulator.py	Simulated zones and sensor readings
ml.py	Random Forest model, training, and persistence filter
api.py	API that receives readings from the ESP32 (port 8000)
send_sensor.py	Sends test sensor data to the API from a terminal
esp32_sensor.ino	ESP32 firmware
check.py	Pipeline self-test
validation.py	Model validation
leak_model.joblib	Trained model
requirements.txt	Python dependencies
start_all.bat	Windows script to start the services
Getting Started
Requirements
Python 3.9 or newer
pip
Install and run
bash
pip install -r requirements.txt

# Optional: retrain the model and regenerate leak_model.joblib
python -c "import ml; ml.train()"

# Optional: pipeline self-test
python check.py

# Start the dashboard
python -m streamlit run app.py

The dashboard opens at http://localhost:8501.

Demo Script (2 minutes)
Open the dashboard. All zones are green and flow and pressure are normal.
In the sidebar, select Zone 3 and click Demand spike (NOT a leak). Flow rises, but no alert is raised. This is the false-alarm filter at work.
Click Inject LEAK. Pressure falls and flow rises. Within about 5 seconds the zone turns red, an alert is logged, and the loss counter starts.
Click Repair leak. The zone returns to green.
Note that a real ESP32 only needs to send the same readings (zone, flow, pressure). The rest of the pipeline stays the same.

Alternatively, click Auto demo to run the whole sequence automatically.

Live Hardware Mode (ESP32)
Start the API (api.py, port 8000).
In the dashboard sidebar, set Data source to Live API (ESP32 plug-in).
Flash esp32_sensor.ino to the ESP32, or use python send_sensor.py to send test readings from a terminal.
Prototype hardware
Component	Role	ESP32 pin
Potentiometer 1	Flow sensor (simulated)	D34 (ADC1)
Potentiometer 2	Pressure sensor (simulated)	D35 (ADC1)
LED (with 220 ohm resistor)	Leak indicator	D13
Buzzer module (low-level trigger)	Leak alarm	D27

D34 and D35 are ADC1 pins, which keeps analog reads reliable while WiFi is active. The buzzer module turns on when its signal pin is driven LOW.

Tech Stack

Python, Pandas, Scikit-learn (Random Forest), Streamlit, Plotly, FastAPI, ESP32 (Arduino)

Roadmap
Real flow and pressure sensor integration on ESP32 nodes
LoRa / GSM gateway for underground installations
PostgreSQL backend and React dashboard
Pilot in a high-risk DMA, then scale zone by zone
Team

Team Dora Ducks

Raj Verma (Team Leader)
Saurabh Tiwari
Vinish Kumar Kashyap
Shorya Mahajan
