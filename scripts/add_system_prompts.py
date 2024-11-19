import jsonlines
import glob
import argparse
import json
import os


def main(args):
    all_dirs = args.dirs
    all_data = []
    for _dir in all_dirs:
        all_data += [
            d
            for d in jsonlines.open("{}/instruction_tuning_examples.jsonl".format(_dir))
        ]

    system_prompt_and_exemplars = json.load(open(args.prompt_path))
    system_prompts = [
        {"role": "system", "content": system_prompt_and_exemplars["intro"]}
    ]
    for ex in system_prompt_and_exemplars["examples"]:
        example_x = ex[0]
        example_y = ex[1]
        system_prompts.append({"role": "example_x", "content": example_x})
        system_prompts.append({"role": "example_y", "content": example_y})

    data_with_system_prompt = []
    for d in all_data:
        d["messages"] = system_prompts + d["messages"]
        data_with_system_prompt.append(d)
    print("Found {} examples".format(len(data_with_system_prompt)))
    target_dir = args.output
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if args.sample != -1:
        data_with_system_prompt = data_with_system_prompt[: args.sample]
    with open("{}/data.jsonl".format(target_dir), "w") as f:
        for d in data_with_system_prompt:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs", nargs="+", help="Directories with instruction tuning data"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="/u/scr/smurty/agents-with-exploration/webarena/agent/prompts/jsons/p_cot_llama.json",
    )
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--output", type=str, default="webarena_data.jsonl")
    args = parser.parse_args()
    main(args)
