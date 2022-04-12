import os
import json
import pandas as pd

errors = {}
errors_by_obj = {}
total_errors = 0
n_false_positives_overriden_by_final_classifier = 0
n_mistakes_by_final_classifier = 0
total_examples = 0
with open("val.jsonl") as f:
    for line in f:
        if len(line.strip()) == 0 or line.startswith("=="): continue
        total_examples += 1
        jline = json.loads(line)
        obj = jline["target"]
        if obj not in errors_by_obj:
            errors_by_obj[obj] = {"false positive": 0, "false negative": 0, "exploration": 0, "correct": 0}
        if jline["success"]:
            errors_by_obj[obj]["correct"] += 1
            continue
        total_errors += 1

        has_false_positive_overriden_by_final_classifier = False
        has_final_classifier_mistake = False


        if jline["failures"][-1].startswith("false_positive"):
            error_type = "false positive"
        elif jline["failures"][-1] == "too_long":
            error_type = "exploration"
            for fail_mode in jline["failures"]:
                if fail_mode.startswith("false_negative: "):
                    error_type = "false negative"
                elif fail_mode.startswith("true_negative"):
                    has_false_positive_overriden_by_final_classifier = True
                elif fail_mode.startswith("false_negative ("):
                    error_type = "false negative"
                    has_final_classifier_mistake = True
        else:
            print(jline)

        if error_type not in errors:
            errors[error_type] = 0
        errors[error_type] += 1
        errors_by_obj[obj][error_type] += 1
        n_false_positives_overriden_by_final_classifier += has_false_positive_overriden_by_final_classifier
        n_mistakes_by_final_classifier += has_final_classifier_mistake

for error_type in errors:
    errors[error_type] /= total_errors

print(errors)
print(f"% false positives overridden by final classifier: {n_false_positives_overriden_by_final_classifier / total_examples}")
print(f"% mistakes by final classifier: {n_mistakes_by_final_classifier / total_examples}")
print("======")

for obj in errors_by_obj:
    total_obj_exs = sum(errors_by_obj[obj].values())
    for error_type in errors_by_obj[obj]:
        # print("    "+error_type + ": " + str(errors_by_obj[obj][error_type]/total_obj_exs))
        errors_by_obj[obj][error_type] = errors_by_obj[obj][error_type] / total_obj_exs
errors_by_obj = pd.DataFrame.from_dict(errors_by_obj)
print(errors_by_obj)
