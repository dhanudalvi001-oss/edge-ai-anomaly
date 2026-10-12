import json, time
import pandas as pd
import paho.mqtt.client as mqtt

cfg = json.load(open("models/config.json"))
FEATURES = cfg["features"]
RATE = 20   # rows per second (real SKAB is 1/s, this speeds up the demo)

df = pd.read_csv("data/SKAB/data/valve1/0.csv", sep=";",
                 index_col="datetime", parse_dates=True)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883)
client.loop_start()

for seq, (_, row) in enumerate(df.iterrows()):
    payload = {
        "seq": seq,
        "ts_sent": time.time(),
        "values": [float(row[f]) for f in FEATURES],
        "label": int(row["anomaly"]),   # used only for evaluation, never by the model
    }
    client.publish("factory/machine1/sensors", json.dumps(payload))
    time.sleep(1 / RATE)

print("Replay finished.")
client.loop_stop()
client.disconnect()