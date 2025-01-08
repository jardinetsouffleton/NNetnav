"""
    Script to run an agent end-to-end on a task.
    It appears similar to evaluation/eval_webarena.py, but has a different purpose.
    There is copied code, but the philosophy is to have separate recipes, instead of a giant script with many flags.
    We also use browsergym.
"""

import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import glob
import openai
from collections import Counter

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    AgentFactory,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)

import browsergym.webarena
from browsergym.webarena import ALL_WEBARENA_TASK_IDS

from nnetnav_utils import get_url
import nnetnav_registry
import webvoyager_registry
from bgym import ExpArgs, EnvArgs
from nnetnav_registry import ALL_OPENENDED_WEBARENA_TASK_IDS, ALL_OPENWEB_TASK_IDS
from webvoyager_registry import ALL_WEBVOYAGER_TASK_IDS

from evaluation_harness import evaluator_router
from agent import InstructionGenerator
from joblib import Parallel, delayed
from agentlab.experiments import args as agentlab_args
from agentlab.experiments import study
from agentlab.experiments.launch_exp import find_incomplete, run_experiments
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.llm.chat_api import (
    SelfHostedModelArgs,
    OpenRouterModelArgs,
    TogetherAIModelArgs,
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


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument("--render", action="store_true", help="Render the browser")
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument("--instruction_generator", action="store_true")
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
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

    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="")

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument("--state_changelog_prompt", type=str, default="")

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
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--inp_task_file", type=str, default="")
    parser.add_argument("--num_instructions", type=int, default=1)
    parser.add_argument(
        "--sample_freq", type=int, default=1, help="If not 1, sample every n tasks"
    )
    parser.add_argument("--use_personas", action="store_true")
    parser.add_argument("--max_steps_exploration", type=int, default=10)
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
    parser.add_argument("--exploration", action="store_true")

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument(
        "--environment_type",
        type=str,
        default="webarena",
        choices=["webarena", "workarena", "miniwob"],
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel joblib jobs"
    )
    parser.add_argument("--agent_name", type=str, default="my_agent")
    parser.add_argument("--use_openrouter", action="store_true")
    parser.add_argument("--use_together_ai", action="store_true")
    parser.add_argument(
        "--data",
        type=str,
        default="nnetnav6k",
        choices=[
            "nnetnav6k",
            "nnetnav1k",
            "nnetnav_ow",
            "webarena_subsampled",
            "webarena",
            "webvoyager",
        ],
    )
    parser.add_argument(
        "--url", type=str, default="", help="If not none, set URLs for webarena"
    )
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


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


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0

    if args.url != "":
        # set WA_SHOPPING, WA_SHOPPING_ADMIN, WA_REDDIT, WA_GITLAB, WA_WIKIPEDIA, WA_MAP, WA_HOMEPAGE using base_url
        base_url = args.url
        os.environ["WA_SHOPPING"] = f"{base_url}:7770/"
        os.environ["WA_SHOPPING_ADMIN"] = f"{base_url}:7780/admin"
        os.environ["WA_REDDIT"] = f"{base_url}:9999"
        os.environ["WA_GITLAB"] = f"{base_url}:8023"
        os.environ["WA_WIKIPEDIA"] = (
            f"{base_url}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        )
        os.environ["WA_MAP"] = f"{base_url}:3000"
        os.environ["WA_HOMEPAGE"] = f"{base_url}:4399"

    test_file_list = []
    changelog_model = None
    if args.use_openrouter:
        # use a reasonably high temperature to get multiple trajectories
        chat_model_args = OpenRouterModelArgs(
            model_name=args.model,
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            temperature=args.temperature,
        )
    elif args.use_together_ai:
        # use a reasonably high temperature to get multiple trajectories
        # TogetherAI uses Turbo postfix for quantized models
        chat_model_args = TogetherAIModelArgs(
            model_name=f"{args.model}-Turbo",
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            temperature=args.temperature,
        )
    else:
        chat_model_args = SelfHostedModelArgs(
            model_name=args.model,
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            backend="vllm",
            n_retry_server=4,
            temperature=args.temperature,
        )

    if args.data == "nnetnav6k":
        env_args_list = [
            EnvArgs(
                task_name=task,
                task_seed=0,
                max_steps=20,
                # task_kwargs={"config_str": json.dumps(conf), "task_id": idx},
            )
            for task in ALL_OPENENDED_WEBARENA_TASK_IDS
        ]
    elif args.data == "nnetnav1k":
        env_args_list = [
            EnvArgs(
                task_name=task,
                task_seed=0,
                max_steps=20,
                # task_kwargs={"config_str": json.dumps(conf), "task_id": idx},
            )
            for task in ALL_OPENENDED_WEBARENA_TASK_IDS[:1000]
        ]
    elif args.data == "nnetnav_ow":
        ### data from nnetnav_openworld
        env_args_list = [
            EnvArgs(
                task_name=task,
                task_seed=0,
                max_steps=20,
            )
            for task in ALL_OPENWEB_TASK_IDS
        ]
    elif args.data == "webarena_subsampled":
        # subsampled evaluation
        env_args_list = [
            EnvArgs(
                task_name=task,
                task_seed=0,
                max_steps=20,
            )
            for idx, task in enumerate(ALL_WEBARENA_TASK_IDS)
            if idx % 8 == 0
        ]
    elif args.data == "webarena":
        env_args_list = [
            EnvArgs(
                task_name=task,
                task_seed=0,
                max_steps=20,
            )
            for task in ALL_WEBARENA_TASK_IDS
        ]
    elif args.data == "webvoyager":
        env_args_list = [
            EnvArgs(
                task_name=task,
                task_seed=0,
                max_steps=20,
            )
            for task in ALL_WEBVOYAGER_TASK_IDS
        ]
    else:
        raise ValueError("Unknown data config")

    if os.path.exists(args.result_dir):
        exp_args_list = find_incomplete(args.result_dir, include_errors=True)
        exp_args_current = Counter([o.status for o in exp_args_list])
        logger.info(f"Current status: {exp_args_current}")
    else:
        agent = AgentFactory(
            flags=FLAGS_GPT_4o,
            chat_model_args=chat_model_args,
            agent_name=args.agent_name,
            args=args,
        )
        exp_args_list = []
        for env_args in env_args_list:
            exp_args = ExpArgs(
                agent_args=agent,
                env_args=env_args,
                logging_level=logging.INFO,
            )
            exp_args_list.append(exp_args)
    logger.info(f"Total {len(exp_args_list)} tasks to run")
    run_experiments(args.n_jobs, exp_args_list, args.result_dir, "joblib", 1)

    # if "debug" not in args.result_dir:
    #     test_file_list = get_unfinished(test_file_list, args.result_dir)
    # if len(test_file_list) == 0:
    #     logger.info("No task left to run")
    # else:
    #     print(f"Total {len(test_file_list)} tasks left")
    #     args.render = False
    #     args.render_screenshot = True
    #     args.save_trace_enabled = True

    #     args.current_viewport_only = True
    #     dump_config(args)

    #     if args.n_jobs == 1:
    #         for config_file in test_file_list:
    #             run(args, agent, config_file)

    #     else:
    #         exp_args = [(args, agent, config_file) for config_file in test_file_list]

    #         def process_arg(exp_arg):
    #             run(exp_arg[0], exp_arg[1], exp_arg[2])

    #         # Parallelize the loop
    #         Parallel(n_jobs=args.n_jobs)(
    #             delayed(process_arg)(exp_arg) for exp_arg in exp_args
    #         )
