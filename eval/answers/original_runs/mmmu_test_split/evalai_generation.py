import json

def read_json_file(input_file, output_file):
    question_count={}
    count = 0
    output_lines = {}
    with open(input_file, "r") as i_f:
        for line in i_f:
            data = json.loads(line)  # Parse each line as a JSON object
            if data['category'] not in question_count:
                question_count[data['category']] = 0
            question_count[data['category']] += 1

            category_string = "test_" + data['category']+ "_" + str(question_count[data['category']])
            
            # output_lineappend = { category_string:  data['answer']}
            # output_line = "{" + category_string + ": " + data['answer'] + "}"
            output_lines[category_string] = data['answer']

            if count == 10:
                break
            else:
                count+= 1
    with open(output_file, "a") as f:  # 'a' mode appends
        json.dump(output_lines, f)
        f.write("\n")  # Ensure each JSON object is on a new line

def main():
    read_json_file("mmmu_norm_test_0.jsonl", "evalai_norm.jsonl")
    read_json_file("mmmu_img_test_0.jsonl", "evalai_img.jsonl")
    read_json_file("mmmu_txt_test_0.jsonl", "evalai_txt.jsonl")

if __name__ == "__main__":
    main()
