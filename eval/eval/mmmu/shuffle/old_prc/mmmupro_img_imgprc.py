import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid

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
    qs = wrong_line["question"]
    # if line["image_2"] is not None:
    #     return None, None, None, None

    # if wrong_line["question_type"] == "multiple-choice":
    qs += " Options:"
    options = re.findall(r"'(.*?)'", wrong_line["options"])
    for i in range(len(options)):
        option = options[i]
        qs += f"\n{chr(ord('A')+i)}. {option}"
    qs += f"\n{args.question_extension}"
    # else:
    #     qs += f"\nAnswer the question using a single word or phrase."

    if line["image_1"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    # remove <image \d> tags
    qs = re.sub(r'<image \d+>', '', qs).strip()

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if line["image_1"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        # image = line["image_1"].convert('RGBA')
        # image = line["image_1"].convert('RGB')
        # image_size = [image.size]
        # image_tensor = process_images([image], image_processor, model_config)
        image = line["image_1"].convert('RGB')
        fake_image = Image.new('RGB', image.size, color='white')
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)
        fake_image_tensor = process_images([fake_image], image_processor, model_config)
        image_tensor = image_tensor[:2] + fake_image_tensor[2:]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt


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
    
    validation_dataset = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    dev_dataset = load_dataset("lmms-lab/MMMU", split="dev")
    # questions = concatenate_datasets([validation_dataset, dev_dataset])
    questions = concatenate_datasets([validation_dataset])
    
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

    wrong_dict = {}
    for line in tqdm(questions):
        # if line["question_type"] == "multiple-choice":
        if line["answer"] not in wrong_dict:
            wrong_dict[line["answer"]] = []
        wrong_dict[line["answer"]].append(line)
        # else:
        #     continue
        # else:
        #     print("We have outlier")
        #     exit()

    idx = -1
    print("Expected length of files:", len(questions))
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    shuffled_questions = questions.shuffle(seed=42)
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue

        # wrong_line = None
        # if line["question_type"] == "multiple-choice":
        wrong_line = random.choice(wrong_dict[line["answer"]])
        wc = 0
        while wrong_line["question"] == line["question"] and wc < 5:
            wrong_line = random.choice(wrong_dict[line["answer"]])
            wc = wc + 1
        # else:
        #     wrong_line = random.choice(questions)
        
        input_ids, image_tensor, image_sizes, prompt = process(line, wrong_line, args, tokenizer, image_processor, model.config)
        if input_ids is None:
            continue
        gt_answer = line["answer"]
        category = line["id"].split('_')[1]
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
        ans_file.write(json.dumps({
            "model_id":model_name,
            "question_id": idx,
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "category": category,
            # "type": line["question_type"]
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
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

