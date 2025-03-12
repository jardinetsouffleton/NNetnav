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
    get_exploration_policy,
    get_reward_model,
    get_trajectory_relabeler,
    get_changelog_model,
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
from dataclasses import dataclass
from typing import Optional

from agent import ExplorationAgentFactory
from bgym import EnvArgs, ExpArgs
from browsergym.core.registration import register_task

# parallelize the exploration episodes
from nnetnav_utils import make_dask_client
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.experiments.launch_exp import run_experiments
from nnetnav_registry import WebArenaOpenEnded, NNetNavOpenEndedTask
from agentlab.experiments.launch_exp import find_incomplete, run_experiments
from agentlab.llm.chat_api import (
    SelfHostedModelArgs,
    OpenRouterModelArgs,
    # TogetherAIModelArgs, # Not implemented
)
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

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


def _get_all_data(all_instructions):
    task_id = 0
    all_data = []
    for goal, config_file in all_instructions:
        with open(config_file, "r") as f:
            _c = json.load(f)
        _c["task_id"] = task_id
        _c["intent"] = goal
        env_type = _c.get("env_type", "webarena")
        if env_type == "webarena":
            _c["start_url"] = get_url(_c["start_url"])
        all_data.append(_c)
        task_id += 1
    return all_data


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
    parser.add_argument("--use_personas", default=True, action="store_true")
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

    parser.add_argument(
        "--seed_dir",
        type=str,
        default="./src/agent/prompts/jsons_openweb/seed_dirs",
    )
    parser.add_argument("--exploration_size_per_seed", type=int, default=2)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=3,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=16000,
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
        default="openweb",
        choices=["webarena", "miniwob", "openweb"],
    )

    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="If this is set, then we will collect data for this url",
    )

    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs to run"
    )
    # in real life, use [4, 8, 12, 16, ..., 40]
    parser.add_argument(
        "--prune_at",
        type=list,
        default=[2],
    )

    args = parser.parse_args()
    return args


def run_nnetnav_exploration_bgym(args):
    """
    Run NNetnav exploration using browsergym + agentlab
    """
    if os.path.exists(args.result_dir):
        all_exps = find_incomplete(args.result_dir, include_errors=True)
    else:
        # first get the prompts for the relabeling, reward and exploration policy
        exploration_policy_prompt_path = get_exploration_policy(
            args, only_path=True, use_llama="llama" in args.model
        )
        reward_model_prompt_path = get_reward_model(args, only_path=True)
        trajectory_labeler_prompt_path = get_trajectory_relabeler(args, only_path=True)
        state_changelog_model_prompt_path = get_changelog_model(args, only_path=True)
        config_list = []
        for f in glob.glob(f"{args.seed_dir}/*.json"):
            if "test" not in f:
                config_list.append(f)
        all_instructions = []
        for _ in range(args.exploration_size_per_seed):
            for config_file in config_list:
                all_instructions.append(("n/a", config_file))

        # chat_model_args = OpenRouterModelArgs(
        #     model_name=f"{args.model}-Turbo",
        #     max_total_tokens=16_384,
        #     max_input_tokens=16_384 - 512,
        #     max_new_tokens=512,
        #     temperature=args.temperature,
        # )
        chat_model_args = CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"]
        all_configs = _get_all_data(all_instructions)
        if args.use_personas:
            if args.environment_type == "webarena":
                with open("src/agent/prompts/jsons/personas.json") as f:
                    personas = json.load(f)
            elif args.environment_type == "miniwob":
                with open("src/agent/prompts/jsons_miniwob/personas.json") as f:
                    personas = json.load(f)
            elif args.environment_type == "openweb":
                with open("src/agent/prompts/jsons_openweb/personas.json") as f:
                    personas = json.load(f)
            else:
                raise ValueError(f"Unknown environment type: {args.environment_type}")
        else:
            personas = None
        # now we create env args corresponding to these configs
        env_args = []
        all_exps = []
        task_names = []
        for idx, _c in enumerate(all_configs):
            if args.environment_type == "webarena":
                gym_id = f"webarena_nnetnav_openended_{idx}"
                task_kwargs = {"config_str": json.dumps(_c), "task_id": idx}
            elif args.environment_type == "openweb":
                gym_id = f"openweb_nnetnav_openended_{idx}"
                task_kwargs = {"start_url": _c["start_url"], "goal": _c["intent"]}
            else:
                raise ValueError(f"Unknown environment type: {args.environment_type}")

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

            agent = ExplorationAgentFactory(
                flags=FLAGS_GPT_4o,
                chat_model_args=chat_model_args,
                args=args,
                task_args=(gym_id, task_kwargs),
                persona_str=persona_str,
                exploration_prompt_constructor_path=exploration_policy_prompt_path,
                change_summarizer_prompt_constructor_path=state_changelog_model_prompt_path,
                trajectory_labeler_prompt_constructor_path=trajectory_labeler_prompt_path,
                outcome_reward_model_prompt_constructor_path=reward_model_prompt_path,
            )
            env_curr = EnvArgs(task_name=gym_id, task_seed=0, max_steps=40)
            exp_curr = ExpArgs(
                agent_args=agent,
                env_args=env_curr,
                logging_level=logging.INFO,
            )
            all_exps.append(exp_curr)

    run_experiments(args.n_jobs, all_exps, args.result_dir, "joblib", 1)


if __name__ == "__main__":
    args = config()
    run_nnetnav_exploration_bgym(args)
