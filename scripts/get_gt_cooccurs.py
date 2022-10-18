import json
import csv
from Stubborn.constants import mpcat40_labels
import numpy as np


roomid2objs = {}
obj_to_room_count = {}
room_annotations = csv.reader(open("room_classification_images/room_annotations.csv"))
room2roomtype = {}
for annotation in room_annotations:
    if len(annotation[2].strip().strip('-')) == 0 or "unclear" in annotation[2]: continue
    room_types = annotation[2].split(',')
    if annotation[0] not in room2roomtype:
        room2roomtype[annotation[0]] = {}
    if annotation[1] not in room2roomtype[annotation[0]]:
        room2roomtype[annotation[0]][annotation[1]] = []
    for room_type in room_types:
        room_type = room_type.strip()
        room2roomtype[annotation[0]][annotation[1]].append(room_type)
possible_objects = mpcat40_labels  #["toilet", "tv_monitor", "chair", "sofa", "plant"]
possible_rooms = [
     'bedroom', 'bathroom', 'closet', 'hallway', 'living room', 'dining room',
     'kitchen', 'entrance', 'basement', 'staircase', 'laundry room',
     'office', 'patio', 'stairs', 'lounge', 'sauna', 'shower room',
     'playroom', 'garage', 'shed', 'storage', 'baby room', 'study',
     "kid's room", 'gym', 'shower', 'attic', 'dressing room', 'pantry']
with open("room_classification_images/saved_val_annotations.txt") as f:
    for line in f:
        line = json.loads(line)
        if line["room_id"] not in room2roomtype[line["scene_id"]]: continue
        room_types = room2roomtype[line["scene_id"]][line["room_id"]]
        # = line["objects"]
        for obj in possible_objects:
            has_obj = obj in line["objects"]
            for present_obj in line["objects"]:
                if obj == "tv_monitor" and "tv" in present_obj:
                    has_obj = True
                    break
                if obj == "sofa" and present_obj == "couch":
                    has_obj = True
                    break
            if obj not in obj_to_room_count:
                obj_to_room_count[obj] = {}
            for room_type in possible_rooms:
                if room_type not in obj_to_room_count[obj]:
                    obj_to_room_count[obj][room_type] = [0,0]
                obj_to_room_count[obj][room_type][1] += 1
            for room_type in room_types:
                obj_to_room_count[obj][room_type][0] += has_obj
gt_cooccurs = []
for obj in possible_objects:
    # print(obj, {room: obj_to_room_count[obj][room][0] / obj_to_room_count[obj][room][1] for room in obj_to_room_count[obj]})
    gt_cooccurs.append([])
    for room in possible_rooms:
        gt_cooccurs[-1].append(obj_to_room_count[obj].get(room, [0,0])[0] / obj_to_room_count[obj].get(room, [0,0])[1])
gt_cooccurs = np.array(gt_cooccurs)
np.save("/raid/lingo/bzl/Stubborn/rednet-finetuning/figures/real_room_obj_plausimplaus_hm3d_cooccurence.npy", gt_cooccurs)

"""
obj_to_room_count = {}
# gt cooccurs
with open("tolmroom_top1_val_onlysamefloor/objroom_cooccurs_nav.jsonl") as f:
    for line in f:
        line = json.loads(line)
        if line["goal"] not in obj_to_room_count:
            obj_to_room_count[line["goal"]] = {}
        for rooms in line["actual rooms"]:
            if rooms == "Unknown": continue
            for room in rooms:
                if room not in obj_to_room_count[line["goal"]]:
                    obj_to_room_count[line["goal"]][room] = 0
                obj_to_room_count[line["goal"]][room] += 1
"""
print(json.dumps(obj_to_room_count, indent=4))