import json
import os


lm_fail_orig_succeed = {}
bad_episodes = open("bad_episodes.txt").readlines()
bad_episodes = [envid.strip() for envid in bad_episodes]

def evaluate(fn, print_prefix, alt_method_envids=set()):
    # alt_method_envids = set()
    accuracy = 0.0
    n_total = 0
    ln = 0
    room_accuracy = 0
    obj_accuracy = {}
    successes = {}
    failures = {}
    with open(fn) as f:
        for line in f:
            if "===" in line: ln = 0; continue
            if len(line.strip()) == 0: ln = 0; continue
            line = json.loads(line)
            if line["target"] not in obj_accuracy:
                obj_accuracy[line["target"]] = [0,0]
            if line["target"] != "tv_monitor": continue
            ln += 1
            # if line["env_id"] in bad_episodes: continue
            # if "XB4GS9ShBRE" not in line["env_id"]: continue
            # if "Nfvxx8J5NCo" not in line["env_id"]: continue
            # if "DYehNKdT76V" not in line["env_id"]: continue
            # if "mL8ThkuaVTM" not in line["env_id"]: continue
            # if "TEEsavR23oF" not in line["env_id"]: continue
            # if "6s7QHgap2fW" not in line["env_id"]: continue
            # if "wcojb4TFT35" not in line["env_id"]: continue
            # if "bxsVRursffK" not in line["env_id"]: continue
            # if "5cdEh9F2hJL" not in line["env_id"]: continue
            # if "DYehNKdT76V" not in line["env_id"]: continue
            line["env_id"].split("_")[0]
            
            if line["metrics"]["success"]:
                successes[line["env_id"]] = f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{ln}/imageme.html"
            else:
                failures[line["env_id"]] = f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{ln}/imageme.html"
            """
            if fn == "tolmroom_top1_val_onlysamefloor/results.jsonl" and line["metrics"]["success"] == 0.0:
                lm_fail_orig_succeed[line["env_id"]] = f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{ln}/imageme.html"
            elif fn == "torandroom_val_onlysamefloor/results.jsonl" and line["metrics"]["success"] == 0.0:
                if line["env_id"] in lm_fail_orig_succeed:
                    del lm_fail_orig_succeed[line["env_id"]]
            """
            alt_method_envids.add(line["env_id"])
            if line["env_id"] in alt_method_envids:
                accuracy += line["metrics"]["success"]
                n_total += 1
                room_accuracy += len(line.get("correctly pred room", [])) > 0
                obj_accuracy[line["target"]][0] += line["metrics"]["success"]
                obj_accuracy[line["target"]][1] += 1
            #if not line["metrics"]["success"]:
            #    #print(line["env_id"] + f" {line['metrics']['success']}: " + f"{os.path.split(fn)[0]}/dump/exp1/episodes/thread_0/eps_{ln}/0-{ln}-Vis-1.png")
            #    #print(line["env_id"] + f" {line['metrics']['success']}: " + f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{ln}/imageme.html")
    print(f"{print_prefix}: {accuracy / n_total}")
    print(f"   room accuracy: {room_accuracy / n_total}")
    print(f"   obj accuracy: " + str({obj: obj_accuracy[obj][0] / obj_accuracy[obj][1] if obj_accuracy[obj][1] != 0 else 0 for obj in obj_accuracy}))
    return alt_method_envids, {"successes": successes, "failures": failures}

alt_method_envids, error_analysis = evaluate("tolmroom_persisttop1_val_onlysamefloor/results.jsonl", "To LM Room (no fallback)")
evaluate("tolmroom_onlystepsthresholdfallback_val_onlysamefloor/results.jsonl", "To LM Room (steps in goal room threshold)", alt_method_envids)
_, error_analysis_lm_room = evaluate("tolmroom_top1_val_onlysamefloor/results.jsonl", "To LM Room (total steps in room + proximity threshold)")
_, error_analysis_gt_room = evaluate("togtroom_nofallback_val_onlysamefloor/results.jsonl", "To GT Room (no fallback)", alt_method_envids)
# evaluate("togtroom_fallback_val_onlysamefloor/results.jsonl", "To GT Room (fallback)", alt_method_envids)
# evaluate("orig_val_onlysamefloor/results.jsonl", "Orig explore", alt_method_envids)
_, error_analysis_random_room = evaluate("torandroom_val_onlysamefloor/results.jsonl", "To rand room", alt_method_envids)
# evaluate("to_gtpriorroom_val_onlysamefloor/results.jsonl", "Gt priors", alt_method_envids)
evaluate("to_gtpriorroom_persisttop1_val_onlysamefloor/results.jsonl", "Gt priors (only top 1)", alt_method_envids)
# evaluate("to_gtpriorroom_nooverride_val_onlysamefloor/results.jsonl", "Gt priors (only top 1)", alt_method_envids)
for env_id in set(error_analysis_gt_room["failures"].keys()).intersection(set(error_analysis_lm_room["successes"].keys())):
    print(env_id, error_analysis_gt_room["failures"][env_id])
