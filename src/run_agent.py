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

from nnetnav_utils import get_url
import webarena_openended
from bgym import ExpArgs, EnvArgs
from webarena_openended import ALL_OPENENDED_WEBARENA_TASK_IDS

from evaluation_harness import evaluator_router
from agent import InstructionGenerator
from joblib import Parallel, delayed
from agentlab.experiments import args as agentlab_args
from agentlab.experiments import study
from agentlab.experiments.launch_exp import find_incomplete, run_experiments
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.llm.chat_api import SelfHostedModelArgs, OpenRouterModelArgs

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


default_oss_llms_args = {
    "n_retry_server": 4,
    "temperature": 0.01,
}


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
    parser.add_argument("--config_dir", type=str, default="config_files")
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
        default=1920,
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
    parser.add_argument(
        "--data", type=str, default="nnetnav6k", choices=["nnetnav6k", "nnetnav1k"]
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


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all([is_equivalent(action, last_action) for action in last_k_actions]):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


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


def run(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file,
    state_changelogger=None,
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    # replace env with browsergym environment..
    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )

    try:
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]
            if "task_id" in _c:
                task_id = _c["task_id"]
            else:
                task_id = os.path.basename(config_file).split(".")[0].split("_")[-1]
                _c["task_id"] = task_id
                # map the start_url env variable to a valid url
                _c["start_url"] = get_url(_c["start_url"])
        render_helper = RenderHelper(_c, args.result_dir, args.action_set_tag)

        # automatically login
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            temp_dir = tempfile.mkdtemp()
            # subprocess to renew the cookie
            subprocess.run(
                [
                    "python",
                    "src/browser_env/auto_login.py",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                ]
            )
            _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
            assert os.path.exists(_c["storage_state"])
            # update the config file
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"
            with open(config_file, "w") as f:
                json.dump(_c, f)

        logger.info(f"[Config file]: {config_file}")
        logger.info(f"[Intent]: {intent}")

        agent.reset(config_file)
        trajectory: Trajectory = []
        obs, info = env.reset(options={"config_file": config_file})
        state_info: StateInfo = {"observation": obs, "info": info}
        if state_changelogger is not None:
            state_info["history"] = "None"
        trajectory.append(state_info)
        history_accum = []
        meta_data = {"action_history": ["None"], "env_type": "webarena"}
        while True:
            early_stop_flag, stop_info = early_stop(
                trajectory, max_steps, early_stop_thresholds
            )
            init_observation = state_info["observation"]["text"]
            if early_stop_flag:
                action = create_stop_action(f"Early stop: {stop_info}")
            else:
                try:
                    action = agent.next_action(trajectory, intent, meta_data=meta_data)
                except ValueError as e:
                    # get the error message
                    action = create_stop_action(f"ERROR: {str(e)}")
            trajectory.append(action)
            action_str = get_action_description(
                action,
                state_info["info"]["observation_metadata"],
                action_set_tag=args.action_set_tag,
                prompt_constructor=(
                    agent.prompt_constructor if isinstance(agent, PromptAgent) else None
                ),
            )
            render_helper.render(action, state_info, meta_data, args.render_screenshot)
            meta_data["action_history"].append(action_str)

            if action["action_type"] == ActionTypes.STOP:
                break

            obs, _, terminated, _, info = env.step(action)
            state_info = {"observation": obs, "info": info}
            final_observation = state_info["observation"]["text"]
            trajectory.append(state_info)
            if terminated:
                # add a action place holder
                trajectory.append(create_stop_action(""))
                break

        if "eval" in _c:
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )
        else:
            score = 1.0

        scores.append(score)

        if score == 1:
            logger.info(f"[Result] (PASS) {config_file}")
        else:
            logger.info(f"[Result] (FAIL) {config_file}")

        if args.save_trace_enabled:
            env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")

    except openai.OpenAIError as e:
        logger.info(f"[OpenAI Error] {repr(e)}")
    except Exception as e:
        logger.info(f"[Unhandled Error] {repr(e)}]")
        import traceback

        # write to error file
        with open(Path(args.result_dir) / "error.txt", "a") as f:
            f.write(f"[Config file]: {config_file}\n")
            f.write(f"[Unhandled Error] {repr(e)}\n")
            f.write(traceback.format_exc())  # write stack trace to file

    render_helper.close()
    env.close()


def prepare(args: argparse.Namespace) -> None:

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [os.path.basename(f).split(".")[0].split("_")[1] for f in result_files]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0
    # prepare(args)

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
    else:
        chat_model_args = SelfHostedModelArgs(
            model_name=args.model,
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            backend="vllm",
            **default_oss_llms_args,
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
    else:
        raise ValueError("Unknown data config")

    if os.path.exists(args.result_dir):
        exp_args_list = find_incomplete(args.result_dir, include_errors=True)
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
