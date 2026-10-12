import json, time, csv, os
from collections import deque
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import tensorflow as tf

MODELS = "models"
cfg = json.load(open(f"{MODELS}/config.json"))
WINDOW, THR = cfg["window"], cfg["ae_threshold"]
scaler = joblib.load(f"{MODELS}/scaler.pkl")

interp = tf.lite.Interpreter(model_path=f"{MODELS}/ae_quant.tflite", num_threads=1)
interp.allocate_tensors()
IN = interp.get_input_details()[0]["index"]
OUT = interp.get_output_details()[0]["index"]

os.makedirs("benchmarks", exist_ok=True)
log_file = open("benchmarks/edge_log.csv", "w", newline="")
log = csv.writer(log_file)
log.writerow(["seq", "label", "score", "pred", "infer_ms", "e2e_ms"])

buf = deque(maxlen=WINDOW)

def on_connect(client, userdata, flags, reason_code, properties):
    print("Edge node connected, waiting for sensor data...")
    client.subscribe("factory/machine1/sensors")

def on_message(client, userdata, msg):
    d = json.loads(msg.payload)
    buf.append(scaler.transform([d["values"]])[0])
    if len(buf) < WINDOW:
        return
    x = np.array(buf, dtype=np.float32).reshape(1, -1)

    t0 = time.perf_counter()
    interp.set_tensor(IN, x)
    interp.invoke()
    recon = interp.get_tensor(OUT)
    infer_ms = (time.perf_counter() - t0) * 1000

    score = float(np.mean((x - recon) ** 2))
    pred = int(score > THR)
    e2e_ms = (time.time() - d["ts_sent"]) * 1000

    log.writerow([d["seq"], d["label"], score, pred, infer_ms, e2e_ms])
    log_file.flush()
    if pred:   # only anomalies leave the edge
        client.publish("factory/machine1/status",
                       json.dumps({"seq": d["seq"], "score": score, "status": "ANOMALY"}))
        print(f"ANOMALY seq={d['seq']} score={score:.4f}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883)
try:
    client.loop_forever()
except KeyboardInterrupt:
    log_file.close()
    print("Stopped, log saved.")