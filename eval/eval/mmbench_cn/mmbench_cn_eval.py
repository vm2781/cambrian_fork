import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import mmh3
import copy

from datasets import load_dataset, concatenate_datasets
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

def hash_image(image):
    image = image.resize((256, 256))
    image_bytes = image.tobytes()
    hash_value = mmh3.hash(image_bytes)
    hex_hash = format(hash_value, '08x')
    return hex_hash


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model_config):
    txt_line = wrong_line1 if args.text_shuffle else line
    if line["hint"] != "nan":
        qs = txt_line["hint"] + "\n" + txt_line["question"] + f" Options:"
    else:
        qs = txt_line["question"] + f" Options:"
    for keys in ["A", "B", "C", "D"]:
        if line[keys] != "nan":
            qs += (f"\n{keys}. "+line[keys])
    # print("BUILD UP QUESTION", qs)

    qs += f"\n{args.question_extension}"
    if line["image"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_hash = ""
    img_line = wrong_line2 if args.image_shuffle else line
    if img_line["image"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image = img_line["image"].convert('RGB')
        image_hash = hash_image(copy.deepcopy(image))
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, image_hash


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Here, text shuffling is", str(args.text_shuffle), "while image shuffling is", str(args.image_shuffle))

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = load_dataset("lmms-lab/MMBench_CN", "default", split="dev")
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")
    
    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    # example_num = 0
    shuffle_questions1 = questions.shuffle(seed=42)
    shuffle_questions2 = questions.shuffle(seed=20)
    with open(chunk_file, "w") as ans_file:
        for line, wrong_line1, wrong_line2 in tqdm(zip(questions, shuffle_questions1, shuffle_questions2), total=len(questions)):
            idx = idx+1
            if idx<valid_chunk[0] or idx>valid_chunk[1]:
                continue
            
            input_ids, image_tensor, image_sizes, image_hash = process(line, wrong_line1, wrong_line2, args, tokenizer, image_processor, model.config)
            gt_answer = line["answer"]
            category = line["category"]
            # l2_category = line["l2-category"]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            source_id = (line["question"]+ " " + image_hash + " " + line["source"])
            ans_file.write(json.dumps({"index": line["index"],
                                    "question": line["question"],
                                    "prediction": outputs,
                                    "gt_answer": gt_answer,
                                    "A":line["A"],
                                    "B":line["B"],
                                    "C":line["C"],
                                    "D":line["D"],
                                    "source_id": source_id,
                                    "model_id": model_name,
                                    "category": category}) + "\n")
            ans_file.flush()
            # if example_num == 3:
            #     break
            # example_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="请直接回答选项字母。")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_shuffle", action='store_true', help="Enable text shuffle")
    parser.add_argument("--image_shuffle", action='store_true', help="Enable image shuffle")
    args = parser.parse_args()

    eval_model(args)