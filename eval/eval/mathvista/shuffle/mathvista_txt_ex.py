import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt

from datasets import load_dataset, concatenate_datasets
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    # get kth chunk out of n chunks cut from lst length
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, wrong_line, args, tokenizer, image_processor, model_config):
    qs = line["query"]

    if wrong_line["decoded_image"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # if line["question_type"] == "multi_choice":
    #     qs += f"\n{args.question_extension}"
    # else:
    #     qs += f"\nAnswer the question using a single word or phrase."
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if wrong_line["decoded_image"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = wrong_line["decoded_image"].convert('RGB')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, qs, image, line["decoded_image"].convert('RGB')


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = load_dataset("AI4Math/MathVista", split="testmini")
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")
    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)

    wrong_dict = {}
    for line in tqdm(questions):
        gt_answer = line["answer"]
        if line["question_type"] == "multi_choice":
            reverse_dict = {}
            for ind, item in enumerate(line["choices"]):
                reverse_dict[item] = chr(ord('A')+ind)
            gt_answer = reverse_dict[gt_answer]
            if gt_answer not in wrong_dict:
                wrong_dict[gt_answer] = []
            wrong_dict[gt_answer].append(line)
        else:
            gt_answer = line["answer"]
            if gt_answer == "2":
                if gt_answer not in wrong_dict:
                    wrong_dict["integer"] = []
                wrong_dict["integer"].append(line)
            elif gt_answer == "[0, 2, 0, 2, 1, 7, 1, 2, 0, 3, 0, 6]":
                if gt_answer not in wrong_dict:
                    wrong_dict["list"] = []
                wrong_dict["list"].append(line)
            elif gt_answer == "1.2":
                if gt_answer not in wrong_dict:
                    wrong_dict["one_dec"] = []
                wrong_dict["one_dec"].append(line)
            elif gt_answer == "0.21":
                if gt_answer not in wrong_dict:
                    wrong_dict["two_dec"] = []
                wrong_dict["two_dec"].append(line)

    example_num = 0
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
        

        wrong_line = None
        if line["question_type"] == "multi_choice":
            gt_answer = line["answer"]
            reverse_dict = {}
            for ind, item in enumerate(line["choices"]):
                reverse_dict[item] = chr(ord('A')+ind)
            gt_answer = reverse_dict[gt_answer]
            wrong_line = random.choice(wrong_dict[gt_answer])
            while wrong_line["question"] == line["question"]:
                wrong_line = random.choice(wrong_dict[gt_answer])
        else:
            gt_answer = line["answer"]
            questions_chosen = None
            if "[" in gt_answer and  "]" in gt_answer:
                questions_chosen = wrong_dict["list"]
            elif len(gt_answer) >= 3 and gt_answer[-3] == ".":
                questions_chosen = wrong_dict["two_dec"]
            elif len(gt_answer) >= 2 and gt_answer[-2] == ".":
                questions_chosen = wrong_dict["one_dec"]
            else:
                questions_chosen = wrong_dict["integer"]
            wrong_line = random.choice(questions_chosen)
            
           

        input_ids, image_tensor, image_sizes, qs, image, right_image = process(line, wrong_line, args, tokenizer, image_processor, model.config)
        if input_ids is None:
            continue
        
        category = line["metadata"]["category"]
        gt_answer = line["answer"]
        if line["question_type"] == "multi_choice":
            reverse_dict = {}
            for ind, item in enumerate(line["choices"]):
                reverse_dict[item] = chr(ord('A')+ind)
            gt_answer = reverse_dict[gt_answer]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_file.write(json.dumps({"model_id":model_name,
                                   "question_id": idx,
                                   "answer": outputs,
                                   "gt_answer": gt_answer,
                                   "category": category,
                                   "type": line["question_type"]}) + "\n")

        if example_num < 3:
            try:
                fig, ax = plt.subplots(figsize=(6, 8))

                ax.axis("off")
                ax.imshow(image)

                caption_text = "Text:", qs, "\nY_Hat:", outputs,"\nTrue_Y:",gt_answer
                plt.figtext(0.5, 0.02, caption_text, wrap=True, horizontalalignment='center', fontsize=12)

                img_dir = "examples/runon/mathvista/txt_" + str(example_num) + "_shuffle.png"
                plt.savefig(img_dir, bbox_inches='tight', dpi=300)

                fig2, ax2 = plt.subplots(figsize=(6, 8))

                ax2.axis("off")
                ax2.imshow(right_image)

                caption_text2 = "Text:", qs,
                plt.figtext(0.5, 0.02, caption_text2, wrap=True, horizontalalignment='center', fontsize=12)

                img_dir2 = "examples/runon/mathvista/txt_" + str(example_num) + "_real.png"
                plt.savefig(img_dir2, bbox_inches='tight', dpi=300)

            except Exception as e:
                print(e)
                example_num -= 1
        else:
            break
        ans_file.flush()
        example_num += 1
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)

