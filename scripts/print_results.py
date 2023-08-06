import json
import os
import numpy as np
from glob import glob
import matplotlib
from matplotlib import pyplot as plt

font = {'family' : 'normal'}
#         'weight' : 'bold',
#         'size'   : 22}
default_font = {}#"fontfamily": "Times New Roman"}
title_font = {
    **default_font,
    "fontsize": 40,
    "fontweight": "bold",
}
axes_label_font = {
    **default_font,
    "fontsize": 30,
    "fontweight": "bold",
}
axes_ticks_font = {
    **default_font,
    "fontsize": 30,
}

matplotlib.rc('font', **font)
plt.rcParams['text.usetex'] = True


# lm_fail_orig_succeed = {}
# bad_episodes = open("bad_episodes.txt").readlines()
# bad_episodes = [envid.strip() for envid in bad_episodes]
def compare_objwise_metrics(m1, m2, k=None):
    # k: specifies printing top-k and botom-k
    relative_obj_metrics = {}
    ordered_objs = []
    for obj in m1:
        if m1.get(obj, None) == None: continue
        if m2.get(obj, None) == None: continue
        if type(m1[obj]) == list:
            if len(m1[obj]) == 0: continue
            m1[obj] = np.array(m1[obj]).mean(0)
        if type(m2[obj]) == list:
            if len(m2[obj]) == 0: continue
            m2[obj] = np.array(m2[obj]).mean(0)
        relative_obj_metrics[obj] = m2[obj] - m1[obj]
        ordered_objs.append(obj)
    ordered_objs.sort(key=relative_obj_metrics.get, reverse=True)
    if k is None:
        ordered_objs_to_print = ordered_objs
    else:
        ordered_objs_to_print = ordered_objs[:k] + ordered_objs[-k:]
    for obj in ordered_objs_to_print:
        print(f"{obj}: {relative_obj_metrics[obj]} ({m2[obj]}-{m1[obj]})")
    return ordered_objs


# def plot_objectwise_metrics(m1, m2, obj_order, figure_fp):
#     x = np.arange(len(m1))  # the label locations
#     width = 0.75  # the width of the bars
#     fig, ax = plt.subplots(figsize = (15, 5))

#     m1_ious = np.array([m1[obj] for obj in obj_order])
#     m2_ious = np.array([m2[obj] for obj in obj_order])
#     mask = (m2_ious - m1_ious) > 0
#     rects1 = ax.bar(x[mask], (m2_ious - m1_ious)[mask], width, color='green') #, label='Original')
#     rects1 = ax.bar(x[~mask],( m2_ious - m1_ious)[~mask], width, color='red') #, label='Original')
#     ax.axhline(y=0, color='k')
#     ax.set_title(r'$\Delta$ Success Rate by object category', **title_font)
#     ax.set_xticks(x)
#     ax.set_xticklabels([obj_name.replace("_", " ") for obj_name in obj_order], rotation=45, ha="right", **axes_ticks_font)
#     ax.tick_params(axis='y', labelsize=axes_ticks_font['fontsize'])
#     # ax.set_yticklabels(range(-0.05, 0.25), **axes_ticks_font)
#     ax.set_ylabel(r'$\Delta$ Success Rate (SR) \\ ($SR_{Socratic} - SR_{Orig}$)', **axes_label_font)
#     fig.tight_layout()
#     plt.savefig(figure_fp)

def plot_objectwise_metrics(m1, m2, obj_order, save_fn):
    x = np.arange(len(m1))  # the label locations
    width = 0.75  # the width of the bars
    fig, ax = plt.subplots(figsize = (6, 2))

    m1_ious = np.array([m1[obj] for obj in obj_order]) * 100
    m2_ious = np.array([m2[obj] for obj in obj_order]) * 100
    mask = (m2_ious - m1_ious) > 0
    rects1 = ax.bar(x[mask], (m2_ious - m1_ious)[mask], width, color='green') #, label='Original')
    rects1 = ax.bar(x[~mask],( m2_ious - m1_ious)[~mask], width, color='red') #, label='Original')
    ax.axhline(y=0, color='k')
    ax.set(xticklabels=[])
    # ax.set_title(r'$\Delta$ Step recall by task category', **title_font)
    # ax.set_xticks(x) #, **axes_ticks_font)
    # ax.set_xticklabels([obj_name[1].replace("_", " ") for obj_name in obj_order], rotation=45, ha="right", **axes_ticks_font)
    ax.tick_params(axis='y', labelsize=axes_ticks_font['fontsize'])
    # ax.set_yticklabels(range(-0.05, 0.25), **axes_ticks_font)
    # ax.set_ylabel(r'$\Delta$ Step Recall (SR) \\ ($SR_{LM\ Prior} - SR_{Orig}$)', **axes_label_font)

    fig.tight_layout()

    plt.savefig(save_fn)


def evaluate(fn, print_prefix, alt_method_envids=None, port=8000):
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
    spl = 0
    obj_spl = {}

    override_accuracies_by_success = {False: [], True: []}
    dir_name = os.path.split(fn)[0]
    with open(fn) as f:
        for line in f:
            if "===" in line: ln = 0; continue
            if len(line.strip()) == 0: ln = 0; continue
            ln += 1
            line = json.loads(line)
            if line["target"] not in obj_accuracy:
                obj_accuracy[line["target"]] = [0,0]
                obj_spl[line["target"]] = [0,0]
            if len(line.get("potential_stop_scenes", {})) > 0:
                for overridden_scene_num in line["potential_stop_scenes"]:
                    overridden_scene = line["potential_stop_scenes"][overridden_scene_num]
                    scene_success = overridden_scene['metrics']['distance_to_goal'] < 0.1
                    try:
                        override_accuracies_by_success[scene_success].append([
                            max(overridden_scene['scores'][0]) if len(overridden_scene['scores'][0]) > 0 else None, overridden_scene['scores'][1]
                        ])
                    except:
                        pass
            
            success = line["metrics"]["success"]
            if not success and len(line.get("extra_goal_positions", [])) > 0:
                success = max([
                    (np.absolute((np.array(line["final_position"]) - np.array(goal_pos["center"]))) < np.array(goal_pos["size"]) + 0.1).all()
                for goal_pos in line["extra_goal_positions"]]) and line['stop_reason'] == "found goal"
            if success:
                successes[line["env_id"]] = f"http://128.30.64.44:{port}/dump/exp1/episodes/thread_0/eps_{line['env_id']}/imageme.html"
            else:
                failures[line["env_id"]] = f"http://128.30.64.44:{port}/dump/exp1/episodes/thread_0/eps_{line['env_id']}/imageme.html"
            if alt_method_envids is None or line["env_id"] in alt_method_envids:
                accuracy += success
                n_total += 1
                room_accuracy += len(line.get("correctly pred room", [])) > 0
                obj_accuracy[line["target"]][0] += success
                obj_accuracy[line["target"]][1] += 1

                spl += line["metrics"]["spl"]
                obj_spl[line["target"]][0] += line["metrics"]["spl"]
                obj_spl[line["target"]][1] += 1
            all_seen_envids.add(line["env_id"])
    # compare_objwise_metrics(obj_accuracy, )

    print(f"{print_prefix}: {accuracy / n_total}")
    print(f"   room accuracy: {room_accuracy / n_total}")
    obj_accuracy = {obj: obj_accuracy[obj][0] / obj_accuracy[obj][1] if obj_accuracy[obj][1] != 0 else 0 for obj in obj_accuracy}
    print(f"   obj accuracy: " + str(obj_accuracy))
    print(f"   SPL: {spl / n_total}")
    print(f"   obj SPL: " + str({obj: obj_spl[obj][0] / obj_spl[obj][1] if obj_spl[obj][1] != 0 else 0 for obj in obj_spl}))
    return all_seen_envids, {"successes": successes, "failures": failures}, obj_accuracy

alt_method_envids, error_analysis_lmprior_combine_override, obj_accuracy_lampp = evaluate("lm_room_classifier_override_switchsteps_sort_all_rooms_newprior/results.jsonl", "To LM Prior Room (rank all rooms, switch by steps + override final decision by LM + classifier)", port=8886)
# _, error_analysis_lmprior_override, obj_accuracy = evaluate("lm_room_switchsteps_lmoverride_sort_all_rooms_newprior/results.jsonl", "To LM Prior Room (rank all rooms, switch by steps + override final decision by LM)", port=8880)
# _, error_analysis_socratic_override, obj_accuracy = evaluate("lm_socratic_override/results.jsonl", "Socratic Rooms (rank all rooms, switch by steps + override final decision by LM + classifier)", port=8800)
_, error_analysis_socratic, obj_accuracy_socratic = evaluate("socratic_rooms_switchsteps/results.jsonl", "Socratic Rooms (rank all rooms, switch by steps)", port=8400)
# _, error_analysis_lmprior = evaluate("lm_room_switchsteps_sort_all_rooms_newprior/results.jsonl", "To LM Prior Room (rank all rooms, switch by steps)", alt_method_envids, port=8000)
_, error_analysis_closest, obj_accuracy_uniform = evaluate("closest_room_switchsteps/results.jsonl", "To Closest Room (switch by steps)", alt_method_envids)
# _, error_analysis_gtprior = evaluate("gt_room_classifier_override_switchsteps_sort_all_rooms_newprior/results.jsonl", "To GT Prior Room (switch by steps)", alt_method_envids)
# _, error_analysis_closest = evaluate("closest_room_switchsteps/results.jsonl", "To Closest Room (switch by steps)", alt_method_envids)
# _, error_analysis_gt = evaluate("gt_room_persist/results.jsonl", "To GT Room (persist search)", alt_method_envids, port=4000)
_, error_analysis_orig, _ = evaluate("orig_val_onlysamefloor/results.jsonl", "Orig explore", alt_method_envids, port=2000)


ordered_objs = compare_objwise_metrics(obj_accuracy_uniform, obj_accuracy_lampp, k=1)
ordered_objs = plot_objectwise_metrics(obj_accuracy_uniform, obj_accuracy_lampp, ordered_objs, "../figures/objnav_zs_lampp.png")
ordered_objs = compare_objwise_metrics(obj_accuracy_uniform, obj_accuracy_socratic, k=1)
ordered_objs = plot_objectwise_metrics(obj_accuracy_uniform, obj_accuracy_socratic, ordered_objs, "../figures/objnav_zs_socratic.png")

# for env_id in set(error_analysis_socratic_override["failures"].keys()).intersection(set(error_analysis_socratic["successes"].keys())):
#     print(env_id, error_analysis_socratic_override["failures"][env_id])
# """