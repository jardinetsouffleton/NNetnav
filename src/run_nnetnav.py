import argparse
import json
import glob
import os
from copy import deepcopy
from nnetnav_utils import (
    LanguagePruning,
    NNetscapeNavigator,
    TrajectoryLabeler,
    RewardModelBatched,
    get_url,
    convert_html_to_jsons,
)
from agent import PromptAgent, InstructionGenerator
from agent.prompts import *
import browsergym.miniwob  # register miniwob tasks as gym environments
import gymnasium as gym
from browser_env import ScriptBrowserEnv

import time
import logging
import random
from browser_env.helper_functions import (
    RenderHelper,
)

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


def _convert_to_shorter_trajectory(demonstration, endpoint_idx):
    """
    demonstration: a dictionary with keys 'messages' and 'intent'
    endpoint_idx: the index of the endpoint to which we want to shorten the trajectory
    """
    shortened_demonstration = deepcopy(demonstration)
    messages = shortened_demonstration["messages"]
    new_messages = []
    m_counter = 0
    for m in messages:
        if "assistant" in m:
            new_messages.append(m)
            m_counter += 1
            # NOTE(smurty): Suppose endpoint_idx = 8: That means we have relabeled s0 a0 s1 a1 s2 a2 s3 a3 s4 a4 s5 a5 s6 a6 s7 a7 s8
            # Since these are 8 state changes. This gives us 8 actions.
            # But we keep the 9th action as well, since this well get post-processed to a [STOP] action.
            if m_counter == endpoint_idx + 1:
                break
        else:
            new_messages.append(m)
    shortened_demonstration["messages"] = new_messages
    return shortened_demonstration


def perform_demonstration_filtering(
    all_rewards,
    all_relabeled_instructions,
    demonstrations_curr,
    result_dir,
    filter_dir,
):
    """
    all_rewards: a dictionary of rewards for every demonstration in demonstrations_curr
    all_relabeled_instructions:

    """
    orig_configs = {}
    # read all jsons of the form config_*.json where * is the example-id
    for f in glob.glob(result_dir + "/instruction_*.json"):
        with open(f) as reader:
            data = json.load(reader)
            example_id = data["task_id"]
            orig_configs["example_{}".format(example_id)] = data
    new_configs = {}
    for key in all_relabeled_instructions:
        if key not in all_rewards:
            continue
        key_example_id = key.split(":")[
            0
        ]  # if all_endpoints, key is of the form example_id:increment
        if key_example_id not in demonstrations_curr:
            continue
        new_configs[key] = deepcopy(orig_configs[key_example_id])
        new_configs[key]["intent"] = all_relabeled_instructions[key]["instruction"]
    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)
    demonstrations_filtered = []

    for key in new_configs:
        example_id = key.split("_")[-1]
        endpoint_idx = int(key.split(":")[-1])
        key_example_id = key.split(":")[0]
        curr_dem = demonstrations_curr[key_example_id]
        # convert into a shorter trajectory
        curr_dem = _convert_to_shorter_trajectory(curr_dem, endpoint_idx)
        curr_dem["intent"] = new_configs[key]["intent"]
        if curr_dem["intent"].lower() == "n/a":
            continue
        if all_rewards[key]["reward"] >= 4:
            demonstrations_filtered.append(curr_dem)
    print("Found {} demonstrations".format(len(demonstrations_filtered)))
    with open(filter_dir + "/filtered_demonstrations_0.json", "w") as f:
        json.dump(demonstrations_filtered, f, indent=4)


def _save_with_goals(all_instructions, result_dir):
    task_id = 0
    all_out_files = []
    all_data = []
    for goal, config_file in all_instructions:
        with open(config_file, "r") as f:
            _c = json.load(f)
        _c["task_id"] = task_id
        _c["intent"] = goal
        _c["start_url"] = get_url(_c["start_url"])
        out_file_curr = f"{result_dir}/instruction_{task_id}.json"
        all_data.append((out_file_curr, _c))
        all_out_files.append(out_file_curr)
        with open(out_file_curr, "w") as f:
            json.dump(_c, f)
        task_id += 1
    return all_out_files, all_data


def config():
    parser = argparse.ArgumentParser(description="Run nnetnav end-to-end")
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_depth", type=int, default=40)
    parser.add_argument("--out_dir", type=str, default="")

    # agent config
    parser.add_argument("--use_personas", action="store_true")
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    parser.add_argument("--model_ig", type=str, default="")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--seed_dir", type=str, default="")
    parser.add_argument("--exploration_size_per_seed", type=int, default=1)
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

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--filter_dir", type=str, default="")

    parser.add_argument(
        "--environment_type",
        type=str,
        default="webarena",
        choices=[
            "webarena",
            "miniwob",
        ],
    )

    args = parser.parse_args()
    return args


def get_changelog_model(args):
    if args.environment_type == "webarena":
        state_changelog_prompt = "src/agent/prompts/jsons/p_state_changelog.json"
    elif args.environment_type == "miniwob":
        state_changelog_prompt = (
            "src/agent/prompts/jsons_miniwob/p_state_changelog.json"
        )
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    llm_config = lm_config.construct_llm_config(args)
    with open(state_changelog_prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        state_changelog_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    changelog_model = InstructionGenerator(
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )

    return changelog_model


def get_trajectory_relabeler(args):
    if args.environment_type == "webarena":
        relabeling_prompt = "src/agent/prompts/jsons/p_instruction_relabel.json"
    elif args.environment_type == "miniwob":
        relabeling_prompt = "src/agent/prompts/jsons_miniwob/p_instruction_relabel.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    llm_config = lm_config.construct_llm_config(args)
    tokenizer = Tokenizer(args.provider, args.model)
    with open(relabeling_prompt, "r") as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    prompt_constructor = eval(constructor_type)(
        relabeling_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    relabeling_model = InstructionGenerator(
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )
    return relabeling_model


def get_reward_model(args):
    llm_config = lm_config.construct_llm_config(args)
    if args.environment_type == "webarena":
        reward_prompt = "src/agent/prompts/jsons/p_reward_lenient.json"
    elif args.environment_type == "miniwob":
        reward_prompt = "src/agent/prompts/jsons_miniwob/p_reward_lenient.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    with open(reward_prompt) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        reward_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    reward_model = InstructionGenerator(
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )
    return reward_model


def get_exploration_policy(args):
    llm_config = lm_config.construct_llm_config(args)
    if args.environment_type == "webarena":
        json_dir = "src/agent/prompts/jsons"
    elif args.environment_type == "miniwob":
        json_dir = "src/agent/prompts/jsons_miniwob"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")

    if args.use_personas:
        zero_shot_policy_prompt = (
            "{}/p_cot_exploration_with_history_persona.json".format(json_dir)
        )
    else:
        zero_shot_policy_prompt = "{}/p_cot_exploration_with_history.json".format(
            json_dir
        )
    with open(zero_shot_policy_prompt) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    prompt_constructor = eval(constructor_type)(
        zero_shot_policy_prompt, lm_config=llm_config, tokenizer=tokenizer
    )
    base_agent = PromptAgent(
        action_set_tag=args.action_set_tag,
        lm_config=llm_config,
        prompt_constructor=prompt_constructor,
    )

    return base_agent


def run_nnetnav_exploration(args):
    # all the prompted LM components in nnetnav
    exploration_policy = get_exploration_policy(args)
    reward_model = get_reward_model(args)
    trajectory_labeler = get_trajectory_relabeler(args)
    state_changelog_model = get_changelog_model(args)
    # for miniwob, we use the browsergym environment
    if args.environment_type == "webarena":
        env = ScriptBrowserEnv(
            headless=True,
            slow_mo=0,
            observation_type="accessibility_tree",
            current_viewport_only=False,
            viewport_size={
                "width": 1280,
                "height": 720,
            },
            save_trace_enabled=False,
            sleep_after_execution=2.0,
        )
    else:
        env = None
    if args.use_personas:
        if args.environment_type == "webarena":
            with open("src/agent/prompts/jsons/personas.json") as f:
                personas = json.load(f)
        elif args.environment_type == "miniwob":
            with open("src/agent/prompts/jsons_miniwob/personas.json") as f:
                personas = json.load(f)
        else:
            raise ValueError(f"Unknown environment type: {args.environment_type}")
    else:
        personas = None
    prune_function = LanguagePruning(reward_model, trajectory_labeler)
    search_obj = NNetscapeNavigator(
        exploration_policy, prune_function=prune_function, max_depth=args.max_depth
    )
    # seed dir is just a config file for loading a web-environment
    # we conduct args.exploration_size_per_seed exploration episodes for each seed
    configs_path = f"{args.result_dir}/test.raw.json"
    if os.path.exists(configs_path):
        all_configs = json.load(open(configs_path, "r"))
    else:
        config_list = []
        for f in glob.glob(f"{args.seed_dir}/*.json"):
            if "test" not in f:
                config_list.append(f)
        all_instructions = []
        for _ in range(args.exploration_size_per_seed):
            for config_file in config_list:
                all_instructions.append(("n/a", config_file))

        _, all_data = _save_with_goals(all_instructions, args.result_dir)
        with open(f"{args.result_dir}/test.raw.json", "w") as f:
            json.dump([v for _, v in all_data], f)
        all_configs = [v for _, v in all_data]

    # here we record some metadata for each exploration episode.
    all_trajectories = {}
    all_history = {}
    all_relabeled = {}
    for _c in all_configs:
        task_id = _c["task_id"]
        # we can resume from a previous run by checking if the history file exists
        conf_file = f"{args.result_dir}/instruction_{task_id}.json"
        history_path = f"{args.result_dir}/history_{task_id}.json"
        if os.path.exists(history_path):
            logger.info(f"Skipping Task: {task_id}")
            all_history[f"example_{task_id}"] = json.load(open(history_path, "r"))
            all_relabeled[f"example_{task_id}"] = json.load(
                open(f"{args.result_dir}/relabeled_instruction_{task_id}.json", "r")
            )
            continue
        render_helper = RenderHelper(_c, args.result_dir, args.action_set_tag)
        # if we are using personas, we sample a persona for each episode to simulate a user
        if personas is not None:
            persona_set = []
            for site in _c["sites"]:
                persona_set += personas[site]
            curr_persona = random.choice(persona_set)
            persona_str = "{}: {}".format(
                curr_persona["persona"], curr_persona["description"]
            )
        else:
            persona_str = ""
        try:
            trajectory_curr, history_curr, relabeled_instruction_dict = search_obj(
                env,
                conf_file,
                persona_str=persona_str,
                state_changelogger=state_changelog_model,
                logger=logger,
                render_helper=render_helper,
            )
            all_trajectories["example_{}".format(task_id)] = trajectory_curr
            all_history["example_{}".format(task_id)] = history_curr
            all_relabeled["example_{}".format(task_id)] = relabeled_instruction_dict
            # write history_curr and relabeled_instruction_dict to file
            with open("{}/history_{}.json".format(args.result_dir, task_id), "w") as f:
                json.dump(history_curr, f)
            with open(
                "{}/relabeled_instruction_{}.json".format(args.result_dir, task_id), "w"
            ) as f:
                json.dump(relabeled_instruction_dict, f)
        except Exception as e:
            logger.info("Error in running subtask: {}".format(e))
            continue
    # dump all_history and all_relabeled to result_dir
    with open("{}/state_changelogs.json".format(args.result_dir), "w") as f:
        json.dump(all_history, f)
    with open("{}/all_relabeled.json".format(args.result_dir), "w") as f:
        json.dump(all_relabeled, f)


if __name__ == "__main__":
    args = config()
    # log the log file
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    with open(os.path.join(args.result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")

    run_nnetnav_exploration(args)
    trajectory_dict = convert_html_to_jsons(
        args.result_dir, f"{args.result_dir}/test.raw.json"
    )

    relabeling_model = get_trajectory_relabeler(args)
    state_changelog_model = get_changelog_model(args)
    reward_module = RewardModelBatched(get_reward_model(args))

    relabeler_module = TrajectoryLabeler(relabeling_model, state_changelog_model)
    relabeled_instructions, state_changelogs = relabeler_module(
        trajectory_dict, args.result_dir, 0, logger=logger, all_endpoints=True
    )
    rewards = reward_module(
        state_changelogs,
        relabeled_instructions,
        args.result_dir,
        logger=logger,
        all_endpoints=True,
    )

    perform_demonstration_filtering(
        rewards,
        relabeled_instructions,
        trajectory_dict,
        args.result_dir,
        args.filter_dir,
    )
