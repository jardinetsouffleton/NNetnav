### Use a language model reward function to evaluate policies


import argparse
from agent.prompts import *
from agent import InstructionGenerator
from nnetnav_utils import convert_html_to_jsons, log_message, convert_to_description


from tqdm import tqdm
import openai
import os
import time
import logging
import random

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


def get_config():
    parser = argparse.ArgumentParser(description="Run LM Reward function")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_dir_2", type=str, default="")
    parser.add_argument("--out_data_path", type=str, default="")

    # agent config
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument("--agent_type", type=str, default="prompt")
    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
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

    parser.add_argument(
        "--environment_type",
        type=str,
        default="webarena",
        choices=["webarena", "workarena", "servicenow"],
    )

    parser.add_argument(
        "--non_oracle",
        action="store_true",
        help="Use non-oracle gpt-4o-mini model without the detailed rubric",
    )
    parser.add_argument(
        "--script_mode",
        choices=["reward_model", "pairwise_reward"],
        default="reward_model",
    )
    args = parser.parse_args()
    return args


def get_pairwise_reward_model(args):
    if args.non_oracle:
        llm_config = lm_config.construct_llm_config(args, model=args.model)
    else:
        llm_config = lm_config.construct_llm_config(args, model="gpt-4o")
    reward_prompt = "agent/prompts/jsons/p_pairwise_reward.json"
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


def get_reward_model(args):
    # use a large model for reward generation
    if args.non_oracle:
        llm_config = lm_config.construct_llm_config(args, model=args.model)
    else:
        llm_config = lm_config.construct_llm_config(args, model="gpt-4o")
    if args.environment_type == "webarena":
        if args.non_oracle:
            reward_prompt = "agent/prompts/jsons/p_reward.json"
        else:
            reward_prompt = "agent/prompts/jsons/p_reward_detailed.json"
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


def get_changelog_model(args):
    if args.environment_type == "webarena":
        state_changelog_prompt = "agent/prompts/jsons/p_state_changelog.json"
    elif args.environment_type == "workarena":
        state_changelog_prompt = "agent/prompts/jsons_workarena/p_state_changelog.json"
    else:
        raise ValueError(f"Unknown environment type: {args.environment_type}")
    # use a small model for changelog generation
    if args.non_oracle:
        llm_config = lm_config.construct_llm_config(args, model=args.model)
    else:
        llm_config = lm_config.construct_llm_config(args, model="gpt-4o-mini")
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


def get_changelogs(log_dir, changelog_model, trajectory_dict, logger=None):
    all_responses = {}
    os.makedirs(f"{log_dir}/tmp_changelogs", exist_ok=True)

    for key in tqdm(trajectory_dict, desc="Getting state changelogs"):
        if os.path.exists(f"{log_dir}/tmp_changelogs/{key}.json"):
            with open(f"{log_dir}/tmp_changelogs/{key}.json", "r") as f:
                all_responses[key] = json.load(f)
            continue
        state_action_state_tuples = []
        state = None
        action = None
        ex = trajectory_dict[key]["messages"]
        for e in ex:
            if "user" in e:
                if state is not None and action is not None:
                    state_action_state_tuples.append(
                        {
                            "init_observation": state,
                            "action": action,
                            "final_observation": e["user"],
                        }
                    )
                state = e["user"]
            if "assistant" in e:
                action = e["assistant"]
        state_action_state_tuples.append(
            {
                "init_observation": state,
                "action": action,
                "final_observation": "Interaction Over!",
            }
        )
        responses = []
        # do not consider trajectories longer than 30 transitions.
        for i in range(len(state_action_state_tuples)):
            try:
                response = changelog_model.generate(
                    state_action_state_tuples[i], meta_data=None
                )
                responses.append(response)
                log_message(logger, f"State Changelog: {response}\n")
            except Exception as e:
                log_message(logger, f"[Exception] {repr(e)}")
        all_responses[key] = responses
        with open(f"{log_dir}/tmp_changelogs/{key}.json", "w") as f:
            json.dump(responses, f, indent=4)
    return all_responses


def get_pairwise_rewards(
    args,
    reward_model,
    instruction2changelog_1,
    instruction2changelog_2,
    curr_dir,
    comparison_name,
    logger=None,
):

    def _post_process(reward_string):
        last_sent = reward_string.split("\n")[-1]
        keyword = "Preference"
        if keyword not in last_sent:
            return 0
        else:
            last_sent = re.sub(keyword, "", last_sent, count=1).strip()
            try:
                reward = float(last_sent)
                return reward
            except ValueError:
                return 0

    reward_file = f"{curr_dir}/rewards_{comparison_name}.json"
    all_rewards = {}
    for instruction in instruction2changelog_1:
        if instruction not in instruction2changelog_2:
            continue
        try:
            c1 = instruction2changelog_1[instruction]
            c2 = instruction2changelog_2[instruction]
            responses_1 = convert_to_description(c1)
            responses_2 = convert_to_description(c2)
            log_message(logger, f"Generating reward for Instruction: {instruction}\n")
            reward = reward_model.generate(
                {
                    "trajectory_1": responses_1,
                    "trajectory_2": responses_2,
                    "instruction": instruction,
                },
                meta_data=None,
            )
            all_rewards[instruction] = {
                "message": reward,
                "reward": _post_process(reward),
            }
            if logger:
                logger.info(reward)
        except openai.OpenAIError as e:
            log_message(logger, f"[OpenAI Error] {repr(e)}")
            continue
        except Exception as e:
            log_message(logger, f"[Exception] {repr(e)}")
            continue
    with open(reward_file, "w") as f:
        json.dump(all_rewards, f, indent=4)
    return all_rewards


def get_rewards(
    args,
    reward_model,
    state_changelogs,
    instructions,
    curr_dir,
    logger=None,
):
    def _post_process(reward_string):
        last_sent = reward_string.split("\n")[-1]
        keyword = "Reward"
        if keyword not in last_sent:
            return 0
        else:
            last_sent = re.sub(keyword, "", last_sent, count=1).strip()
            try:
                reward = float(last_sent)
                return reward
            except ValueError:
                return 0

    if args.non_oracle:
        reward_file = f"{curr_dir}/rewards.json"
    else:
        reward_file = f"{curr_dir}/rewards_oracle.json"
    if os.path.exists(reward_file):
        all_rewards = json.load(open(reward_file, "r"))
        return [r["reward"] for r in all_rewards]
    all_rewards = {}
    for key in state_changelogs:
        try:
            responses = convert_to_description(state_changelogs[key])
            intent = instructions[key]
            log_message(logger, f"Generating reward for Instruction: {intent}\n")
            reward = reward_model.generate(
                {"trajectory": responses, "instruction": intent}, meta_data=None
            )
            all_rewards[key] = {
                "message": reward,
                "reward": _post_process(reward),
            }
            if logger:
                logger.info(reward)
        except openai.OpenAIError as e:
            log_message(logger, f"[OpenAI Error] {repr(e)}")
            continue
        except Exception as e:
            log_message(logger, f"[Exception] {repr(e)}")
            continue
    with open(reward_file, "w") as f:
        json.dump(all_rewards, f, indent=4)
    return all_rewards


def main_prm(args):
    pairwise_reward_model = get_pairwise_reward_model(args)
    changelog_dict = json.load(open("{}/changelogs.json".format(args.data_dir), "r"))
    changelog_dict_model_2 = json.load(
        open("{}/changelogs.json".format(args.data_dir_2), "r")
    )

    trajectory_dict_1 = json.load(open("{}/json_dump.json".format(args.data_dir), "r"))
    trajectory_dict_2 = json.load(
        open("{}/json_dump.json".format(args.data_dir_2), "r")
    )

    instruction2changelog_1 = {
        v["intent"]: changelog_dict[k] for k, v in trajectory_dict_1.items()
    }

    instruction2changelog_2 = {
        v["intent"]: changelog_dict_model_2[k] for k, v in trajectory_dict_2.items()
    }

    m1 = args.data_dir.split("/")[-1]
    m2 = args.data_dir_2.split("/")[-1]
    comparison_name = "{}_{}".format(m1, m2)
    get_pairwise_rewards(
        args,
        pairwise_reward_model,
        instruction2changelog_1,
        instruction2changelog_2,
        args.data_dir,
        comparison_name,
        logger,
    )


def main_rm(args):
    reward_model = get_reward_model(args)
    changelog_model = get_changelog_model(args)
    if os.path.exists(
        "{}/filtered_parsed_with_retroactive_stop_action.json".format(args.data_dir)
    ):
        data = json.load(
            open(
                "{}/filtered_parsed_with_retroactive_stop_action.json".format(
                    args.data_dir
                ),
                "r",
            )
        )
        all_trajectory_dict = {}
        task_id_c = 0
        for d in data:
            retroactive_messages = d["retroactive_reasoning"][:-1] + [
                "Let's think step-by-step. I have finished my tasks. In summary, the next action I will perform is ```{}```".format(
                    d["stop_action"]
                )
            ]
            messages = d["messages"]
            new_messages = []
            curr_action_id = 0
            for i, m in enumerate(messages):
                if "assistant" in m:
                    new_messages.append(
                        {"assistant": retroactive_messages[curr_action_id]}
                    )
                    curr_action_id += 1
                else:
                    new_messages.append(m)
            # override the stop action message
            new_messages[-1] = {"assistant": retroactive_messages[-1]}
            all_trajectory_dict["example_{}".format(task_id_c)] = {
                "intent": d["intent"],
                "messages": new_messages,
            }
            task_id_c += 1

    elif not os.path.exists("{}/json_dump.json".format(args.data_dir)):
        config_file = "{}/test.raw.json".format(args.data_dir)
        all_trajectory_dict = convert_html_to_jsons(args.data_dir, config_file)
    else:
        all_trajectory_dict = json.load(
            open("{}/json_dump.json".format(args.data_dir), "r")
        )
    if not os.path.exists("{}/changelogs.json".format(args.data_dir)):
        changelog_dict = get_changelogs(
            args.data_dir, changelog_model, all_trajectory_dict, logger
        )
        with open("{}/changelogs.json".format(args.data_dir), "w") as f:
            json.dump(changelog_dict, f, indent=4)
    else:
        changelog_dict = json.load(
            open("{}/changelogs.json".format(args.data_dir), "r")
        )
    instructions = {k: v["intent"] for k, v in all_trajectory_dict.items()}
    all_rewards = get_rewards(
        args,
        reward_model,
        changelog_dict,
        instructions,
        args.data_dir,
        logger,
    )


if __name__ == "__main__":
    args = get_config()
    if args.script_mode == "reward_model":
        main_rm(args)
    elif args.script_mode == "pairwise_reward":
        main_prm(args)
    else:
        raise ValueError(f"Unknown script mode: {args.script_mode}")
