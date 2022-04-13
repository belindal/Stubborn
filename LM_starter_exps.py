import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import matplotlib.pyplot as plt
import json
import math
import csv
from tqdm import tqdm
import os

import openai
# openai.organization = "org-8jXkUFeFDJqIpWtgvtpuPjwm"
openai.api_key = "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"


model_distributions = {"T5": {}, "GPT2": {}, "GPT3": {}}


def normalize_distribution(input_list):
    return [item / sum(input_list) for item in input_list]

def exponentiate(inputs):
    outputs = {}
    for item in inputs:
        if inputs[item] > 0:
            outputs[item] = -inputs[item]
        else:
            outputs[item] = inputs[item]
        outputs[item] = math.exp(outputs[item])
    return outputs

def convert_to_probs(obj2class2logprob, do_exponentiate=False):
    obj2class2prob = {}
    for obj in obj2class2logprob:
        obj2class2prob[obj] = {}
        if do_exponentiate:
            obj2class2prob[obj] = exponentiate(obj2class2logprob[obj])
        total_p_mass = sum(obj2class2logprob[obj].values())
        for cl in obj2class2logprob[obj]:
            obj2class2prob[obj][cl] = obj2class2logprob[obj][cl] / total_p_mass
    return obj2class2prob

def plot_distr(room_order, distributions, title=None, do_normalize=False):
    for model in distributions:
        rooms = [room for room in room_order if room in distributions[model]]
        ys = [distributions[model][room] for room in rooms]
        if do_normalize:
            ys = normalize_distribution(ys)
        plt.plot(rooms, ys, label=model)
    xs = list(range(len(rooms)))
    plt.xticks(rooms, rotation=90)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def score_obj_class_gpt3(template, objects, classes, engine="text-davinci-001", save_file=None):
    if save_file is not None and os.path.exists(save_file):
        obj2class2logprob = json.load(open(save_file))
    else:
        obj2class2logprob = {}
        for obj in tqdm(objects):
            obj2class2logprob[obj] = {}
            for cl in classes:
                if obj == cl: continue
                output = openai.Completion.create(
                #search_model="ada",
                engine=engine,
                #examples=[["",""]],
                prompt=template.format(obj=obj, cl=cl),
                max_tokens=0,
                logprobs=0,
                echo=True,
                #labels=["bag", "piano", "babysitter"],
                )
                prefix = template.format(obj=obj, cl="|").split("|")[0].strip()
                for token_position in range(len(output['choices'][0]['logprobs']['tokens'])):
                    if ''.join(output['choices'][0]['logprobs']['tokens'][:token_position]).strip() == prefix:
                        break
                # token_position = output['choices'][0]['logprobs']['tokens'].index(' '+cl.split(' ')[0])
                #print(output['choices'][0]['logprobs']['token_logprobs'][token_position:])
                obj2class2logprob[obj][cl] = sum(output['choices'][0]['logprobs']['token_logprobs'][token_position:])
        if save_file is not None:
            with open(save_file, "w") as wf:
                json.dump(obj2class2logprob, wf)
    obj2class2prob = convert_to_probs(obj2class2logprob, do_exponentiate=True)
    return obj2class2prob


def score_obj_class(objects, classes, tokenizer, model, template):
    # loss_fn = torch.nn.LogSoftmax(-1)
    obj2class2logprob = {}
    for obj in tqdm(objects):
        obj2class2logprob[obj] = {}
        for cl in classes:
            if cl == obj: continue
            input_str = template.format(obj=obj, cl=cl)
            input_ids = tokenizer(input_str, return_tensors="pt", return_offsets_mapping=True).input_ids.to('cuda')
            if type(model) == T5ForConditionalGeneration:
                labels = t5_tokenizer(f"<extra_id_0> {cl}<extra_id_1>", return_tensors="pt").input_ids.to('cuda')
            elif type(model) == GPT2LMHeadModel:
                labels = input_ids
            else:
                raise NotImplementedError()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            obj2class2logprob[obj][cl] = torch.exp(-loss).item()
    obj2class2prob = convert_to_probs(obj2class2logprob, do_exponentiate=False)
    return obj2class2prob


correlation_type = "objroomobj"  # "objobj", "objroomobj"
GPT3_SAVE_FILE = f"LM_results/gpt3_{correlation_type}_correlations.json"
ALL_SAVE_FILE = f"LM_results/all_models_{correlation_type}_correlations.json"

if correlation_type == "objobj":
    # objects = ["bed", "toilet", "TV", "sink", "stove", "sofa", "floor"]
    # classes = ["bathroom", "bedroom", "kitchen", "living room"]
    # objects = ["chair", "couch", "potted plant", "bed", "toilet", "TV"]
    objects = ["chair", "sofa", "potted plant", "bed", "toilet", "TV"]
    classes = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest of drawers', 'plant', 'sink', 'toilet', 'stool', 'towel', 'TV monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym equipment', 'seating', 'clothes']
    t5_template = "The {obj} is near the <extra_id_0>"

    # objects = ["chair", "couch", "potted plant", "bed", "toilet", "TV"]
    #classes = ["bathroom", "bedroom", "kitchen", "living room"]
    #objects = ["bed", "toilet", "TV", "sink", "stove", "sofa", "floor"]
    gpt2_template = "The {obj} is near the {cl}."

    # objects_room = ["chair", "couch", "potted plant", "bed", "toilet", "TV"]
    # objects_obj = ["chair", "sofa", "potted plant", "bed", "toilet", "TV"]
    # classes_obj = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest of drawers', 'plant', 'sink', 'toilet', 'stool', 'towel', 'TV monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym equipment', 'seating', 'clothes']
    gpt3_template = "The {obj} is near the {cl}."
elif correlation_type.startswith("objroom"):
    objects = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest of drawers', 'plant', 'sink', 'toilet', 'stool', 'towel', 'TV monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym equipment', 'seating', 'clothes']
    classes = ["bathroom", "bedroom", "kitchen", "living room", "office", "dining room", "hallway"]
    t5_template = "The {obj} is in the <extra_id_0>"
    gpt2_template = "The {obj} is in the {cl}."
    gpt3_template = "The {obj} is in the {cl}."
else:
    raise NotImplementedError


if os.path.exists(ALL_SAVE_FILE):
    model_distributions = json.load(open(ALL_SAVE_FILE))
else:
    t5_tokenizer = T5TokenizerFast.from_pretrained("t5-large")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-large").to('cuda')
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda')
    print("Loaded models")
    print("Scoring T5")
    model_distributions["T5"] = score_obj_class(objects, classes, t5_tokenizer, t5_model, t5_template)
    print("Scoring GPT2")
    model_distributions["GPT2"] = score_obj_class(objects, classes, gpt2_tokenizer, gpt2_model, gpt2_template)
    print("Scoring GPT3")
    model_distributions["GPT3"] = score_obj_class_gpt3(gpt3_template, objects, classes, engine="text-davinci-001", save_file=GPT3_SAVE_FILE)
    print("Done!")

    with open(ALL_SAVE_FILE, "w") as wf:
        json.dump(model_distributions, wf)

do_normalize = True
if correlation_type == "objroomobj":
    # now compute obj-obj correlations from obj-room correlations
    from_objs = ["chair", "sofa", "plant", "bed", "toilet", "TV monitor"]
    to_objs = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest of drawers', 'plant', 'sink', 'toilet', 'stool', 'towel', 'TV monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym equipment', 'seating', 'clothes']
    new_model_distributions = {}
    for model in model_distributions:
        new_model_distributions[model] = {}
        for obj in from_objs:
            new_model_distributions[model][obj] = {}
            for obj2 in to_objs:
                if obj2 == obj: continue
                new_model_distributions[model][obj][obj2] = 0
                for room in model_distributions[model][obj]:
                    new_model_distributions[model][obj][obj2] += model_distributions[model][obj][room] * model_distributions[model][obj2][room]
                if model == "GPT3": breakpoint()
    model_distributions = new_model_distributions
    do_normalize = False

print(new_model_distributions)
# for obj in objects:
#     plot_distr(classes, {model: model_distributions[model][obj] for model in model_distributions}, title=obj, do_normalize=do_normalize)