"""
Script to run ORM on traces, to get reward annotations for webarena. 

Can also be used with a "non_future_conditioned" judge model as well 
"""

from joblib import Parallel, delayed
import argparse
import glob
import os
import pickle, gzip
import numpy as np
from browsergym.experiments.loop import yield_all_exp_results

from agent import LMModule
from nnetnav_utils import get_reward_model, get_changelog_model, convert_to_description


from agentlab.llm.chat_api import SelfHostedModelArgs
from agent.prompts import PassivePromptConstructor
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.agent_configs import (
    CHAT_MODEL_ARGS_DICT,
    FLAGS_GPT_4o,
)
from tqdm import tqdm

default_oss_llms_args = {
    "n_retry_server": 4,
    "temperature": 0.01,
}
from agentlab.llm.chat_api import SelfHostedModelArgs, OpenRouterModelArgs
import json
from llms import (
    lm_config,
)
from llms.tokenizers import Tokenizer


def get_prompt_constructor(args, prompt_path):
    llm_config = lm_config.construct_llm_config(args)
    with open(prompt_path) as f:
        constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
    tokenizer = Tokenizer(args.provider, args.model)
    # for some reason this import is needed otherwise joblib parallelization fails
    # TODO (smurty): figure out why this is needed
    from agent.prompts import PassivePromptConstructor

    prompt_constructor = eval(constructor_type)(
        prompt_path, lm_config=llm_config, tokenizer=tokenizer
    )
    return prompt_constructor


def config():
    parser = argparse.ArgumentParser(description="Run ORM on traces")

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
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

    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs to run"
    )
    parser.add_argument(
        "--orig_trace_dir",
        type=str,
        help="Directory containing orig traces (if we want to do pairwise comparison)",
        default="",
    )

    parser.add_argument(
        "--output_prefix", type=str, help="Output prefix for saving results"
    )
    parser.add_argument("--use_openrouter", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nnetnav_subset", type=int, default=10000)
    parser.add_argument(
        "--environment_type",
        type=str,
        default="webarena",
        choices=["webarena", "workarena", "miniwob"],
    )

    args = parser.parse_args()
    return args


def get_all_exp_args(root_dir):
    all_exp_dirs = [f for f in glob.glob(f"{root_dir}/*") if os.path.isdir(f)]
    all_exp_out = []
    for exp_dir in all_exp_dirs:
        if os.path.exists(f"{exp_dir}/exp_args.pkl"):
            exp_args = pickle.load(open(f"{exp_dir}/exp_args.pkl", "rb"))
            all_exp_out.append(exp_args)
    return all_exp_out


def get_trajectory_summary(orig_exp_dir, change_summarizer, redo=False):
    previous_steps = [
        pickle.load(gzip.open(f, "rb")) for f in glob.glob(f"{orig_exp_dir}/step_*.gz")
    ]
    if os.path.exists(f"{orig_exp_dir}/summary.pkl") and not redo:
        summary = pickle.load(open(f"{orig_exp_dir}/summary.pkl", "rb"))
        return summary
    else:
        previous_steps_indexed = {s.step: s for s in previous_steps}
        summary = []
        all_steps = [idx for idx in range(len(previous_steps))]
        for s in all_steps:
            # if last step, then we don't have a next step
            if s + 1 not in previous_steps_indexed:
                break
            curr_observation = previous_steps_indexed[s].obs["axtree_txt"]
            next_observation = previous_steps_indexed[s + 1].obs["axtree_txt"]
            curr_action = previous_steps_indexed[s].action
            change_summary_curr = change_summarizer(
                {
                    "init_observation": curr_observation,
                    "final_observation": next_observation,
                    "action": curr_action,
                }
            )
            summary.append(change_summary_curr)
        with open(f"{orig_exp_dir}/summary.pkl", "wb") as f:
            pickle.dump(summary, f)
        return summary


def run_orm(exp_arg):
    # not that these flags don't really matter because we are directly using the axtree_txt objects...
    FLAGS_GPT_4o_webarena = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=False,
            use_action_history=True,
            use_think_history=False,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=False,
            use_som=False,
            extract_visible_tag=True,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            multi_actions=False,
            action_set="webarena",
            long_description=False,
            individual_examples=False,
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=True,
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=True,
        enable_chat=False,
        max_prompt_tokens=None,
        be_cautious=True,
        extra_instructions=None,
    )

    if args.use_openrouter:
        # use a reasonably high temperature to get multiple trajectories
        chat_model_args = OpenRouterModelArgs(
            model_name=args.model,
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            temperature=args.temperature,
        ).make_model()
    else:
        chat_model_args = SelfHostedModelArgs(
            model_name=args.model,
            max_total_tokens=16_384,
            max_input_tokens=16_384 - 512,
            max_new_tokens=512,
            backend="vllm",
            **default_oss_llms_args,
        ).make_model()

    reward_prompt = get_prompt_constructor(args, get_reward_model(args, only_path=True))
    summarizer_prompt = get_prompt_constructor(
        args, get_changelog_model(args, only_path=True)
    )

    change_summarizer = LMModule(
        chat_model_args, FLAGS_GPT_4o_webarena, summarizer_prompt, max_retry=3
    )
    reward_model = LMModule(
        chat_model_args, FLAGS_GPT_4o_webarena, reward_prompt, max_retry=3
    )

    orig_exp_dir = exp_arg.exp_dir
    try:
        summary = get_trajectory_summary(orig_exp_dir, change_summarizer)
        # get the instruction
        instruction = pickle.load(
            gzip.open(f"{orig_exp_dir}/goal_object.pkl.gz", "rb")
        )[0]["text"]
        trajectory = convert_to_description([s["output"] for s in summary])
        if os.path.exists(f"{orig_exp_dir}/reward.pkl"):
            reward = pickle.load(open(f"{orig_exp_dir}/reward.pkl", "rb"))
        else:
            reward = reward_model(
                {"instruction": instruction, "trajectory": trajectory}
            )
            with open(f"{orig_exp_dir}/reward.pkl", "wb") as f:
                pickle.dump(reward, f)
    except:
        print(f"Error in {orig_exp_dir}")
        reward = {"think": "", "reward": -1}
    print(reward)
    return reward


def execute_in_parallel(exp_set):
    """
    exp_set is a list of lists, where each inner list is a tuple of (exp_arg, orig_trajectory)
    """
    delayed_funcs = []

    def get_task(exp_arg_curr):
        return delayed(run_orm)(exp_arg_curr)

    for exp_arg_curr in exp_set:
        delayed_funcs.append(get_task(exp_arg_curr))

    compute(delayed_funcs)


if __name__ == "__main__":
    args = config()

    # define the reward model and change summarizer
    FLAGS_GPT_4o_webarena = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=False,
            use_action_history=True,
            use_think_history=False,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=False,
            use_som=False,
            extract_visible_tag=True,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            multi_actions=False,
            action_set="webarena",
            long_description=False,
            individual_examples=False,
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=True,
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=True,
        enable_chat=False,
        max_prompt_tokens=None,
        be_cautious=True,
        extra_instructions=None,
    )
    chat_model_args = CHAT_MODEL_ARGS_DICT["azure/gpt-4o-mini-2024-07-18"]

    # batched version
    trace2dir_orig = {}
    for fname in glob.glob(f"{args.orig_trace_dir}/*"):
        if os.path.isdir(fname):
            env_info = pickle.load(open(f"{fname}/exp_args.pkl", "rb")).env_args
            task_name = env_info.task_name
            task_id = int(env_info.task_name.split("_")[-1])
            if task_id < args.nnetnav_subset:
                trace2dir_orig[task_id] = fname
        if args.debug and len(trace2dir_orig) > 10:
            break

    exp_set = []
    for task_id in tqdm(trace2dir_orig):
        try:
            exp_arg = pickle.load(
                open("{}/exp_args.pkl".format(trace2dir_orig[task_id]), "rb")
            )
            exp_set.append(exp_arg)
        except Exception as e:
            print(f"Error in task {task_id}: {e}")
    print("Running regular ORM for multiple tasks: ", len(exp_set))
    if args.n_jobs == 1:
        for exp_arg in exp_set:
            run_orm(exp_arg)
    else:
        with Parallel(n_jobs=args.n_jobs) as parallel:
            parallel(delayed(run_orm)(exp_arg) for exp_arg in exp_set)
