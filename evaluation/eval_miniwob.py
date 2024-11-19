"""Script to run end-to-end evaluation on MiniWob++ using browsergym"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
import gymnasium as gym
import browsergym.miniwob  # register miniwob tasks as gym environments


from agent import (
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    ActionTypes,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import (
    RenderHelper,
)
from browsergym.utils.obs import flatten_dom_to_str
from agent import InstructionGenerator

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument("--task_name", type=str, default="miniwob.choose-date-nodelay")
    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )

    # lm config
    parser.add_argument("--action_set_tag", type=str, default="id_accessibility_tree")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )
    parser.add_argument("--state_changelog_prompt", type=str, default="")
    # example config
    parser.add_argument("--test_start_idx", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=20)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    return args


def create_config(response, config_file, out_dir):
    """
    Replaces the intent in config_file with response
    """

    with open(config_file) as f:
        data = json.load(f)
    data["intent"] = response
    out_file = "{}/{}".format(out_dir, os.path.basename(config_file))
    with open(out_file, "w") as f:
        json.dump(data, f, indent=4)


def convert_to_description(changelogs):
    """
    returns a natural language description of all the changes to states
    """
    i = 0
    descriptions = []
    for log in changelogs:
        cstr = "Step: " + str(i) + "\n"
        cstr += log
        descriptions.append(cstr)
        i += 1
    return "\n\n".join(descriptions)


def test(
    args,
    agent,
    env,
    seed_id,
    config_dict,
    result_dir,
    task_name,
    state_changelogger=None,
) -> float:
    scores = []
    max_steps = args.max_steps

    # do not actually use the action_set_tag for miniwob...
    render_helper = RenderHelper(config_dict, result_dir, action_set_tag="")
    trajectory: Trajectory = []
    obs, info = env.reset(seed=seed_id)
    obs["text"] = flatten_dom_to_str(
        obs["dom_object"],
        filter_visible_only=True,
        extra_properties=obs["extra_element_properties"],
        with_clickable=True,
    )
    intent = obs["goal"]
    info["observation_metadata"] = {"text": {"obs_nodes_info": obs["axtree_object"]}}
    state_info: StateInfo = {"observation": obs, "info": info}
    if state_changelogger is not None:
        state_info["history"] = "None"
    trajectory.append(state_info)

    meta_data = {
        "action_history": ["None"],
        "env_type": "miniwob++",
    }
    history_accum = []
    all_rewards = 0.0
    while True:
        if len(trajectory) >= 2 * max_steps:
            stop_info = f"Reach max steps {max_steps}"
            action = create_stop_action(f"Early stop: {stop_info}")
        else:
            try:
                action = agent.next_action(trajectory, intent, meta_data=meta_data)
            except ValueError as e:
                # get the error message
                action = create_stop_action(f"ERROR: {str(e)}")

        init_observation = state_info["observation"]["text"]
        trajectory.append(action)
        if action["action_type"] == ActionTypes.STOP:
            break
        render_helper.render(action, state_info, meta_data, render_screenshot=True)
        obs, reward, terminated, _, info = env.step(action["parsed_response"])
        all_rewards += reward
        logger.info(info)
        if obs["last_action_error"]:
            action_str = f"ERROR: {obs['last_action_error']}"
        else:
            action_str = action["parsed_response"]
        obs["text"] = flatten_dom_to_str(
            obs["dom_object"],
            filter_visible_only=True,
            extra_properties=obs["extra_element_properties"],
            with_clickable=True,
        )

        state_info = {"observation": obs, "info": info}
        final_observation = state_info["observation"]["text"]
        logger.info("Action: {}".format(action_str))
        if state_changelogger is not None:
            state_changelogger_inp = {
                "init_observation": init_observation,
                "action": action_str,
                "final_observation": final_observation,
            }
            curr_changelog = state_changelogger.generate(
                state_changelogger_inp, meta_data=None
            )
            print(curr_changelog)
            history_accum.append(curr_changelog)
            state_info["history"] = convert_to_description(history_accum)

        meta_data["action_history"].append(action_str)
        trajectory.append(state_info)
        if terminated:
            # add a action place holder
            trajectory.append(create_stop_action(""))
            break

    logger.info(f"[Result]: {all_rewards}")
    # write reward and seed id to result_dir/result_{seed_id}.json
    with open(f"{result_dir}/result_{seed_id}.json", "w") as f:
        json.dump(
            {
                "reward": all_rewards,
                "seed_id": seed_id,
                "task_id": task_name,
                "intent": intent,
            },
            f,
            indent=4,
        )

    return all_rewards


if __name__ == "__main__":
    args = config()
    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_start_idx + args.num_seeds
    task_name = args.task_name
    agent = construct_agent(args)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.state_changelog_prompt:
        llm_config = lm_config.construct_llm_config(args)
        with open(args.state_changelog_prompt, "r") as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.state_changelog_prompt, lm_config=llm_config, tokenizer=tokenizer
        )
        changelog_model = InstructionGenerator(
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )
    else:
        changelog_model = None

    if task_name == "all":
        tasks = [
            "miniwob.book-flight-nodelay",
            "miniwob.choose-date-nodelay",
            "miniwob.click-checkboxes-soft",
            "miniwob.email-inbox",
            "miniwob.login-user",
            "miniwob.navigate-tree",
            "miniwob.phone-book",
            "miniwob.use-autocomplete-nodelay",
        ]
    else:
        tasks = [task_name]
    for task in tasks:
        logger.info(f"[Task]: {task}")
        env = gym.make("browsergym/{}".format(task))
        result_dir = "{}/{}".format(args.result_dir, task)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        all_rewards = []
        for i in range(st_idx, ed_idx):
            if os.path.exists("{}/result_{}.json".format(result_dir, i)):
                _c = json.load(open("{}/result_{}.json".format(result_dir, i)))
                all_rewards.append(_c["reward"])
                continue
            config_dict = {"task_name": task, "task_id": i}
            reward = test(
                args, agent, env, i, config_dict, result_dir, task, changelog_model
            )
            all_rewards.append(reward)
        avg_reward = sum(all_rewards) / len(all_rewards)
        logger.info(f"[Avg Reward]: {avg_reward} for task {task}")
