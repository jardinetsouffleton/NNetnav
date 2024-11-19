# Code to convert web-agent trajectories into instruction tuning dataimport argparse
import argparse
import json, subprocess, tempfile, os
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.actions import create_id_based_action
import glob
from browser_env import ScriptBrowserEnv
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)

from bs4 import BeautifulSoup
from llms.tokenizers import Tokenizer
from agent.prompts import *
import jsonlines

from agent.prompts import *

ipath = "src/agent/prompts/jsons/p_cot_llama.json"
lm_config = None
tokenizer = None
DATASET = "webarena"
prompt = CoTPromptConstructor(ipath, lm_config, tokenizer)


def add_system_prompt(args, sft_base_path):
    all_data = [d for d in jsonlines.open(sft_base_path)]

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
    target_dir = args.output_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open("{}/data.jsonl".format(target_dir), "w") as f:
        for d in data_with_system_prompt:
            f.write(json.dumps(d) + "\n")


def get_instruction_tuning_example(
    instruction_template, intent, observation, previous_action, message, url=None
):
    """
    convert intent, metadata, previous action and message into instruction tuning example
    """
    to_fill = {
        "observation": observation,
        "objective": intent,
        "past_actions": previous_action,
        "url": url,
    }
    user_instruction = instruction_template.format(**to_fill)
    assistant_output = message
    return [
        {"role": "user", "content": user_instruction},
        {"role": "assistant", "content": assistant_output},
    ]


def main(args):
    # not all environments need a stop action
    instruction_template = json.load(open(args.prompt_path, "r"))["template"]
    if os.path.exists(
        "{}/filtered_parsed_with_retroactive_stop_action.json".format(
            args.nnetnav_dem_dir
        )
    ):
        with open(
            "{}/filtered_parsed_with_retroactive_stop_action.json".format(
                args.nnetnav_dem_dir
            ),
            "r",
        ) as f:
            demonstrations = json.load(f)
    elif os.path.exists(
        "{}/filtered_parsed_with_retroactive.json".format(args.nnetnav_dem_dir)
    ):
        with open(
            "{}/filtered_parsed_with_retroactive.json".format(args.nnetnav_dem_dir), "r"
        ) as f:
            demonstrations = json.load(f)
    else:
        raise ValueError("Bad input.")
    all_instruction_tuning_examples = []
    train_on = args.train_on
    for demonstration in demonstrations:
        if train_on != "all" and demonstration["sites"][0] not in train_on:
            continue
        observations = []
        actions = []
        dem_id = demonstration["task_id"]
        for m in demonstration["messages"]:
            if "user" in m:
                observations.append(m["user"])
            else:
                actions.append(m["assistant"])

        if args.only_actions:
            actions = [
                "In summary the next action I will perform is ```{}```".format(action)
                for action in demonstration["parsed_actions"]
            ]
        actions = demonstration["retroactive_reasoning"]
        # not all environments need a stop action
        if args.add_stop_action:
            # replace the last action with the stop action. Please see _convert_to_shorter_trajectory in run_lgs.py for context
            # for why this is the correct thing to do
            actions = actions[:-1] + [
                "Let's think step-by-step. I have finished my tasks. In summary, the next action I will perform is ```{}```".format(
                    demonstration["stop_action"]
                )
            ]
        dem_size = len(observations) - 1
        with open(
            "{}/render_states/render_{}.html".format(args.nnetnav_dem_dir, dem_id), "r"
        ) as f:
            render_state = f.read()
            soup = BeautifulSoup(render_state, "html.parser")
            previous_actions = [
                obv.get_text() for obv in soup.find_all("div", {"class": "prev_action"})
            ]
        print(demonstration["intent"])
        previous_actions_curr = []
        for idx, obs in enumerate(observations):
            webpage = obs.split("observation:")[-1].strip()
            url = obs.split("observation:")[0].strip().split("URL:")[-1].strip()
            url = prompt.map_url_to_real(url)
            previous_actions_curr.append(previous_actions[idx])
            if DATASET == "workarena":
                instruction_tune_example = get_instruction_tuning_example(
                    demonstration["intent"],
                    webpage,
                    previous_actions_curr[-1],
                    actions[idx],
                    url,
                )
            else:
                _with_steps = [
                    "{}: {}".format(jdx + 1, a)
                    for jdx, a in enumerate(previous_actions_curr)
                ]
                instruction_tune_example = get_instruction_tuning_example(
                    instruction_template,
                    demonstration["intent"],
                    webpage,
                    "\n".join(_with_steps),
                    actions[idx],
                    url,
                )
            all_instruction_tuning_examples.append(
                {
                    "dataset": "webarena",
                    "id": "example_{}".format(dem_id),
                    "messages": instruction_tune_example,
                }
            )

    return all_instruction_tuning_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get Playwright traces for nnetnav demonstrations"
    )
    parser.add_argument(
        "--nnetnav_dem_dir",
        type=str,
        help="Directory where parsed nnetnav demonstrations are stored",
    )
    parser.add_argument("--dem_size_threshold", type=int, default=10000)
    parser.add_argument("--dem_size_min", type=int, default=0)

    parser.add_argument("--only_actions", action="store_true")
    parser.add_argument("--add_stop_action", action="store_true")
    parser.add_argument("--train_on", type=str, default="all", nargs="+")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/agent/prompts/jsons/p_cot_llama_action_history.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuning/data/processed/nnetnav_data",
        help="Directory to store output for supervised fine-tuning",
    )

    args = parser.parse_args()

    all_messages_set = main(args)
    print("Created {} supervised finetuning examples".format(len(all_messages_set)))
    if args.train_on == "all":
        out_file = "{}/sft_examples.jsonl".format(args.nnetnav_dem_dir)
    else:
        out_file = "{}/sft_examples_{}.jsonl".format(
            args.nnetnav_dem_dir, "_".join(args.train_on)
        )
    with open(out_file, "w") as f:
        for example in all_messages_set:
            f.write(json.dumps(example) + "\n")

    # now add system prompts
    add_system_prompt(args, out_file)
