import glob
import json

with open("data/train/gaze_ann/default.json") as f:
    data = json.load(f)
    print(type(data))
    print(len(data))
    
    print(data.keys())
    print(len(data["items"]))
    print(data["items"][0])

    print(data["items"][0].keys())
    print(len(data["items"][2]["annotations"]))
