import playwright.sync_api
import json

# we use a global playwright instance
_PLAYWRIGHT = None


def _set_global_playwright(pw: playwright.sync_api.Playwright):
    global _PLAYWRIGHT
    _PLAYWRIGHT = pw


def _get_global_playwright():
    global _PLAYWRIGHT
    if not _PLAYWRIGHT:
        pw = playwright.sync_api.sync_playwright().start()
        _set_global_playwright(pw)

    return _PLAYWRIGHT


# register the open-ended task
from browsergym.core.registration import register_task
from .webvoyager_task import WebVoyagerTask


# get the current path, and use that to load configs located in the same directory
def get_configs():
    import os

    current_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(current_path, "webvoyager_data/WebVoyager_data.jsonl")
    return [json.loads(s) for s in open(config_path).readlines()]


configs = get_configs()
ALL_WEBVOYAGER_TASK_IDS = []

for idx, _c in enumerate(configs):
    task_id = _c["id"]
    # replace all " " with "_"
    task_id = task_id.replace(" ", "_")
    gym_id = f"webvoyager_{task_id}"
    register_task(
        gym_id,
        WebVoyagerTask,
        task_kwargs={
            "web_name": _c["web_name"],
            "id": _c["id"],
            "goal": _c["ques"],
            "start_url": _c["web"],
        },
    )
    ALL_WEBVOYAGER_TASK_IDS.append(gym_id)
