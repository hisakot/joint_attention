import glob
import json

ROLL = ["NotHuman", "Surgeon", "Assistant", "Anesthesiology", "ScrubNurse", "CirculationNurse", "Visitor"]

def load_mmpose_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        instances = data["instance_info"]

        return instances

if __name__ == '__main__':
    data_dir = "data/train"
    mmpose_paths = glob.glob(data_dir + "/roll_ann/mmpose/*.json")
    mmpose_paths.sort()
    datas = []
    for file in mmpose_paths:
        instances = load_mmpose_json(file)
        datas.extend(instances)

    rolls = [0] * len(ROLL)
    instance_num = 0
    for data in datas:
        frame_id = data["frame_id"]
        instances = data["instances"]

        for instance in instances:
            try:
                rolls[ROLL.index(instance["roll"])] += 1
                instance_num += 1
            except KeyError:
                pass
    print(ROLL)
    print(rolls)
    print(instance_num)
    print("----------")
    print([roll / instance_num for roll in rolls])
    print([1 / roll for roll in rolls])
    weights = []
    weight_sum = 0
    for roll in rolls:
        weights.append(1 / roll)
        weight_sum += 1 / roll
    weights = [w / weight_sum for w in weights]
    print(weights)
