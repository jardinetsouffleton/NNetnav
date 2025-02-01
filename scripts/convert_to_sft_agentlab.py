"""
    New version of convert_to_sft.py that uses agentlab infra
"""

import argparse
import json
import os
from browser_env.env_config import URL_MAPPINGS
from agentlab.llm.llm_utils import count_tokens
from tqdm import tqdm
from llms.tokenizers import Tokenizer

system_chat_message_webarena = {
    "role": "system",
    "content": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous actions: These are all the action you have performed. It may be helpful to track your progress.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\nHomepage:\nIf you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. You are strictly forbidden from issuing a goto action to a URL that is not on the homepage.\n5. Generate the action in the correct format. Start by reasoning about the current situation. End with \"In summary, the next action I will perform is\" phrase, followed by action inside ``````. For example, \"Let's think step-by-step. Given the current state, I need to click on the like button which has id 1234. In summary, the next action I will perform is ```click [1234]```\".\n6. Issue stop action when you think you have achieved the objective. Don't generate anything after stop. \n\nHere are some example outputs for some random tasks:\n1. Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```\n2. Let's think step-by-step. This page has a search box whose ID is [164]. According to the nominatim rule of openstreetmap, I can search for the restaurants near a location by \"restaurants near\". I can submit my typing by pressing the Enter afterwards. In summary, the next action I will perform is ```type [164] [restaurants near CMU] [1]```",
}

system_chat_message_openweb = {
    "role": "system",
    "content": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous actions: These are all the action you have performed. It may be helpful to track your progress.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. Generate the action in the correct format. Start by reasoning about the current situation. End with \"In summary, the next action I will perform is\" phrase, followed by action inside ``````. For example, \"Let's think step-by-step. Given the current state, I need to click on the like button which has id 1234. In summary, the next action I will perform is ```click [1234]```\".\n5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop. \n\nHere are some example outputs for some random tasks:\n1. Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```\n2. Let's think step-by-step. This page has a search box whose ID is [164]. According to the nominatim rule of openstreetmap, I can search for the restaurants near a location by \"restaurants near\". I can submit my typing by pressing the Enter afterwards. In summary, the next action I will perform is ```type [164] [restaurants near CMU] [1]```",
}


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


def map_url_to_real(url: str) -> str:
    """Map the urls to their real world counterparts, for webarena"""
    for i, j in URL_MAPPINGS.items():
        if i in url:
            url = url.replace(i, j)
    return url


def main(
    prompt_path,
    nnetnav_dem_dir,
    train_on,
    environment_type,
    model_name,
    exp_name,
    max_obs_length=16000,
):
    instruction_template = json.load(open(prompt_path, "r"))["template"]
    if os.path.exists(
        "{}/iltered_parsed_with_retroactive_stop_action.json".format(nnetnav_dem_dir)
    ):
        with open(
            "{}/iltered_parsed_with_retroactive_stop_action.json".format(
                nnetnav_dem_dir
            ),
            "r",
        ) as f:
            demonstrations = json.load(f)
    else:
        raise ValueError("Bad input.")

    if environment_type == "webarena":
        system_chat_message = system_chat_message_webarena
    else:
        system_chat_message = system_chat_message_openweb

    all_instruction_tuning_examples = []
    tokenizer = Tokenizer(provider="vllm", model_name=model_name)
    for demonstration in tqdm(demonstrations):
        # if the length of the intent is more than 100 words, skip
        if len(demonstration["intent"].split()) > 100:
            continue
        if train_on != "all" and demonstration["sites"][0] not in train_on:
            continue
        observations = demonstration["obs"]

        parsed_actions = demonstration["parsed_actions"]
        actions = []

        for orig_action, with_reasoning in zip(
            parsed_actions[:-1], demonstration["retroactive_reasoning"][:-1]
        ):
            if with_reasoning == "":
                actions.append(
                    "Let's think step-by-step. My next action is ```{}```".format(
                        orig_action
                    )
                )
            else:
                actions.append(with_reasoning)

        actions.append(demonstration["stop_action"])
        previous_actions = ["None"] + demonstration["action_descriptions"]
        previous_actions_curr = []
        for idx, obs in enumerate(observations):
            webpage = obs
            # use a tokenizer to make the size of obs <= 16k, since this is our context window limit
            first_line = obs.split("\n")[0]
            webpage = tokenizer.decode(tokenizer.encode(webpage)[:max_obs_length])  # type: ignore[arg-type]
            # the pattern is ".... url='<url>'". We need to extract the url
            try:
                url = first_line.split("url=")[1][1:-1]
            except:
                url = "not_found"
            if environment_type == "webarena":
                url = map_url_to_real(url)

            previous_actions_curr.append(previous_actions[idx])
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
            chat_message = [system_chat_message] + instruction_tune_example

            # need to follow the following format for instruction tuning of llama
            full_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{}\n<|eot_id|>".format(
                chat_message[0]["content"],
                chat_message[1]["content"],
            )

            output = (
                "<|start_header_id|>assistant<|end_header_id|>\n{}\n<|eot_id|>".format(
                    chat_message[-1]["content"]
                )
            )

            task_name = demonstration["task_id"]
            n_tokens = count_tokens(full_prompt, model_name)
            if n_tokens > 100000:
                # too big a context for training
                continue
            output_curr = {
                "dataset": f"webarena_{exp_name}",
                "id": task_name,
                "output": output,
                "task_name": task_name,
                "prompt": full_prompt,
                "messages": chat_message,
                "n_tokens": n_tokens,
            }

            all_instruction_tuning_examples.append(output_curr)

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
    parser.add_argument("--train_on", type=str, default="all", nargs="+")

    parser.add_argument(
        "--environment_type",
        type=str,
        default="webarena",
        choices=["webarena", "openweb"],
        help="Environment type, webarena or openweb",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="Model name for token count",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="llm",
        help="Experiment name for tracking",
    )

    args = parser.parse_args()

    if args.environment_type == "webarena":
        prompt_path = "src/agent/prompts/jsons/p_cot_llama_action_history.json"
    else:
        prompt_path = "src/agent/prompts/jsons_openweb/p_cot_llama_action_history.json"

    outputs = main(
        prompt_path,
        args.nnetnav_dem_dir,
        args.train_on,
        args.environment_type,
        args.model_name,
        args.exp_name,
    )

    all_task_names = [o["task_name"] for o in outputs]
    print("Created {} supervised finetuning examples".format(len(outputs)))

    train_size = int(0.9 * len(all_task_names))
    val_size = int(0.1 * len(all_task_names))
    train_tasks = set(all_task_names[:train_size])
    val_tasks = set(all_task_names[train_size:])

    train_outputs = [o for o in outputs if o["task_name"] in train_tasks]
    val_outputs = [o for o in outputs if o["task_name"] in val_tasks]
    test_outputs = val_outputs

    DATA_DUMP_DIR = "/u/scr/smurty/agents-with-exploration/public/nnetnav_datasets"
    os.makedirs(f"{DATA_DUMP_DIR}/{args.exp_name}", exist_ok=True)

    with open(
        f"{DATA_DUMP_DIR}/{args.exp_name}/train.jsonl", "w", encoding="utf-8"
    ) as f:
        for item in train_outputs:
            f.write(json.dumps(item) + "\n")

    with open(f"{DATA_DUMP_DIR}/{args.exp_name}/val.jsonl", "w", encoding="utf-8") as f:
        for item in val_outputs:
            f.write(json.dumps(item) + "\n")

    with open(
        f"{DATA_DUMP_DIR}/{args.exp_name}/test.jsonl", "w", encoding="utf-8"
    ) as f:
        for item in test_outputs:
            f.write(json.dumps(item) + "\n")

    # take the val and test tasks and write them to eval_tasks.txt
    with open(f"{DATA_DUMP_DIR}/{args.exp_name}/eval_tasks.txt", "w") as f:
        for task in val_tasks:
            f.write(task + "\n")

    # also write the full data
    with open(f"{DATA_DUMP_DIR}/{args.exp_name}/all.jsonl", "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")
