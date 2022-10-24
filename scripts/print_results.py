import json
import os
import numpy as np


lm_fail_orig_succeed = {}
bad_episodes = open("bad_episodes.txt").readlines()
bad_episodes = [envid.strip() for envid in bad_episodes]

def evaluate(fn, print_prefix, alt_method_envids=None):
    # if alt_method_envids is None:
    #     alt_method_envids = set()
    accuracy = 0.0
    n_total = 0
    ln = 0
    room_accuracy = 0
    obj_accuracy = {}
    successes = {}
    failures = {}
    all_seen_envids = set()
    with open(fn) as f:
        for line in f:
            if "===" in line: ln = 0; continue
            if len(line.strip()) == 0: ln = 0; continue
            ln += 1
            line = json.loads(line)
            if line["target"] not in obj_accuracy:
                obj_accuracy[line["target"]] = [0,0]
            # if line["target"] != "chair": continue
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
            # if line["env_id"] != "81_Dd4bFSTQ8gi_chair": continue
            # line["env_id"].split("_")[0]
            # if line["env_id"] not in {'47_5cdEh9F2hJL_chair', '36_Nfvxx8J5NCo_chair', '41_XB4GS9ShBRE_chair', '62_cvZr5TUy5C5_chair', '50_bxsVRursffK_chair', '3_q3zU7Yy5E5s_chair', '32_TEEsavR23oF_chair', '60_5cdEh9F2hJL_chair', '74_QaLdnwvtxbs_chair', '4_5cdEh9F2hJL_chair', '58_q3zU7Yy5E5s_chair', '61_Dd4bFSTQ8gi_chair', '33_mv2HUxq3B53_chair', '98_5cdEh9F2hJL_chair', '94_DYehNKdT76V_chair', '35_q3zU7Yy5E5s_chair', '18_bxsVRursffK_chair', '81_p53SfW6mjZe_chair', '0_6s7QHgap2fW_chair', '95_DYehNKdT76V_chair', '91_q3zU7Yy5E5s_chair', '58_QaLdnwvtxbs_chair', '74_Dd4bFSTQ8gi_chair', '74_5cdEh9F2hJL_chair', '60_Dd4bFSTQ8gi_chair', '0_Dd4bFSTQ8gi_chair', '56_cvZr5TUy5C5_chair', '89_mv2HUxq3B53_chair', '52_5cdEh9F2hJL_chair', '57_qyAac8rV8Zk_chair', '34_DYehNKdT76V_chair', '85_XB4GS9ShBRE_chair', '89_p53SfW6mjZe_chair', '12_cvZr5TUy5C5_chair', '29_mv2HUxq3B53_chair', '32_5cdEh9F2hJL_chair', '51_Nfvxx8J5NCo_chair', '84_5cdEh9F2hJL_chair', '89_Dd4bFSTQ8gi_chair', '87_q3zU7Yy5E5s_chair', '37_Dd4bFSTQ8gi_chair', '6_p53SfW6mjZe_chair', '81_DYehNKdT76V_chair', '83_p53SfW6mjZe_chair', '64_svBbv1Pavdk_chair', '36_Dd4bFSTQ8gi_chair', '51_bxsVRursffK_chair', '40_5cdEh9F2hJL_chair', '76_5cdEh9F2hJL_chair', '2_q3zU7Yy5E5s_chair', '88_qyAac8rV8Zk_chair', '31_DYehNKdT76V_chair', '95_q3zU7Yy5E5s_chair', '94_Nfvxx8J5NCo_chair', '45_6s7QHgap2fW_chair', '62_q3zU7Yy5E5s_chair', '1_q3zU7Yy5E5s_chair', '47_QaLdnwvtxbs_chair', '90_bxsVRursffK_chair', '96_qyAac8rV8Zk_chair', '97_Dd4bFSTQ8gi_chair'}:
            #     continue
            
            success = line["metrics"]["success"]
            if not success and len(line.get("extra_goal_positions", [])) > 0:
                success = max([
                    (np.absolute((np.array(line["final_position"]) - np.array(goal_pos["center"]))) < np.array(goal_pos["size"]) + 0.1).all()
                    # (np.absolute((np.array(line["final_position"]) - np.array(goal_pos["center"]))) < np.array(goal_pos["size"]) / 2 + 0.1).all()
                for goal_pos in line["extra_goal_positions"]]) and line['stop_reason'] == "found goal"
                # success1 = max([
                #     (np.absolute((np.array(line["final_position"]) - np.array(goal_pos["center"]))) < (np.array(goal_pos["size"]) / 2) + 0.5).all()
                #     # (np.absolute((np.array(line["final_position"]) - np.array(goal_pos["center"]))) < np.array(goal_pos["size"]) / 2 + 0.1).all()
                # for goal_pos in line["extra_goal_positions"]]) and line['stop_reason'] == "found goal"
                # if fn == "lm_room_switchsteps/results.jsonl" and line['stop_reason'] == "found goal" and not success:
                #     print(f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{line['env_id']}/imageme.html")
                #     for goal_pos in line["extra_goal_positions"]: 
                #         print((np.absolute((np.array(line["final_position"]) - np.array(goal_pos["center"]))) - np.array(goal_pos["size"]) / 2))
                #     breakpoint()
            if success:
                successes[line["env_id"]] = f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{line['env_id']}/imageme.html"
            else:
                failures[line["env_id"]] = f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{line['env_id']}/imageme.html"
            """
            if fn == "tolmroom_top1_val_onlysamefloor/results.jsonl" and line["metrics"]["success"] == 0.0:
                lm_fail_orig_succeed[line["env_id"]] = f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{ln}/imageme.html"
            elif fn == "torandroom_val_onlysamefloor/results.jsonl" and line["metrics"]["success"] == 0.0:
                if line["env_id"] in lm_fail_orig_succeed:
                    del lm_fail_orig_succeed[line["env_id"]]
            """
            # print(line["env_id"] + "\t" + line["stop_reason"])
            if alt_method_envids is None or line["env_id"] in alt_method_envids:
                accuracy += success
                n_total += 1
                room_accuracy += len(line.get("correctly pred room", [])) > 0
                obj_accuracy[line["target"]][0] += success
                obj_accuracy[line["target"]][1] += 1
            all_seen_envids.add(line["env_id"])
            #if not line["metrics"]["success"]:
            #    #print(line["env_id"] + f" {line['metrics']['success']}: " + f"{os.path.split(fn)[0]}/dump/exp1/episodes/thread_0/eps_{ln}/0-{ln}-Vis-1.png")
            #    #print(line["env_id"] + f" {line['metrics']['success']}: " + f"http://128.30.64.44:8000/dump/exp1/episodes/thread_0/eps_{ln}/imageme.html")
    print(f"{print_prefix}: {accuracy / n_total}")
    print(f"   room accuracy: {room_accuracy / n_total}")
    print(f"   obj accuracy: " + str({obj: obj_accuracy[obj][0] / obj_accuracy[obj][1] if obj_accuracy[obj][1] != 0 else 0 for obj in obj_accuracy}))
    return all_seen_envids, {"successes": successes, "failures": failures}

# alt_method_envids = set()
alt_method_envids, error_analysis_lmprior = evaluate("lm_room_switchsteps/results.jsonl", "To LM Prior Room (switch by steps)")
_, error_analysis_lmprior_lmoverride = evaluate("lm_room_switchsteps_lmoverride/results.jsonl", "To LM Prior Room (switch by steps + override final decision by LM)", alt_method_envids)
_, error_analysis_gtprior = evaluate("gtprior_room_switchsteps/results.jsonl", "To GT Prior Room (switch by steps)", alt_method_envids)
_, error_analysis_closest = evaluate("closest_room_switchsteps/results.jsonl", "To Closest Room (switch by steps)", alt_method_envids)
_, error_analysis_gt = evaluate("gt_room_persist/results.jsonl", "To GT Room (persist search)", alt_method_envids)
_, error_analysis_orig = evaluate("orig_val_onlysamefloor/results.jsonl", "Orig explore", alt_method_envids)
# print(set(error_analysis_lmprior["failures"].keys()).intersection(set(error_analysis_orig["successes"].keys())))
# for env_id in set(error_analysis_lmprior["failures"].keys()).intersection(set(error_analysis_orig["successes"].keys())):
#     print(env_id, error_analysis_lmprior["failures"][env_id])

"""
# alt_method_envids, error_analysis = evaluate("tolmroom_persisttop1_val_onlysamefloor/results.jsonl", "To LM Room (no fallback)")
alt_method_envids, error_analysis_lm_room_onlygoalsteps = evaluate("tolmroom_onlystepsthresholdfallback_val_onlysamefloor/results.jsonl", "To LM Room (steps in goal room threshold)")
_, error_analysis_lm_room_proximity_totsteps = evaluate("tolmroom_top1_val_onlysamefloor/results.jsonl", "To LM Room (total steps in room + proximity threshold)")
_, error_analysis_gt_room = evaluate("togtroom_nofallback_val_onlysamefloor/results.jsonl", "To GT Room (no fallback)", alt_method_envids)
# evaluate("togtroom_fallback_val_onlysamefloor/results.jsonl", "To GT Room (fallback)", alt_method_envids)
# evaluate("orig_val_onlysamefloor/results.jsonl", "Orig explore", alt_method_envids)
_, error_analysis_random_room = evaluate("torandroom_val_onlysamefloor/results.jsonl", "To rand room", alt_method_envids)
# evaluate("to_gtpriorroom_val_onlysamefloor/results.jsonl", "Gt priors", alt_method_envids)
evaluate("to_gtpriorroom_persisttop1_val_onlysamefloor/results.jsonl", "Gt priors (only top 1)", alt_method_envids)
# evaluate("to_gtpriorroom_nooverride_val_onlysamefloor/results.jsonl", "Gt priors (only top 1)", alt_method_envids)
for env_id in set(error_analysis_lm_room_onlygoalsteps["failures"].keys()).intersection(set(error_analysis_lm_room_proximity_totsteps["successes"].keys())):
    print(env_id, error_analysis_lm_room_onlygoalsteps["failures"][env_id])
"""