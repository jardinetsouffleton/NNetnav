import json


def main(args):
    configs = json.load(open("config_files/test.raw.json"))
    configs_to_keep = []
    for config in configs:
        if config["task_id"] % 8 == 0:
            configs_to_keep.append(config)

    if args.model_based:
        acc_out = json.load(open(f"{args.result_dir}/rewards_oracle.json"))
        acc_dict = {}
        for k in acc_out:
            task_id = k.split("_")[-1]
            acc_dict[task_id] = acc_out[k]["reward"]
    else:
        acc_file = [
            l.strip()
            for l in open(f"{args.result_dir}/merged_log.txt")
            if "PASS" in l or "FAIL" in l
        ]
        acc_dict = {}
        for l in acc_file:
            task_id = l.split("/")[-1].split(".")[0]
            is_pass = "PASS" in l
            acc_dict[task_id] = is_pass

    per_domain_acc = {}
    for config in configs_to_keep:
        task_id = config["task_id"]
        domain = config["sites"][0]
        if domain == "wikipedia":
            domain = config["sites"][1]
        if domain not in per_domain_acc:
            per_domain_acc[domain] = []
        if str(task_id) in acc_dict:
            per_domain_acc[domain].append(acc_dict[str(task_id)])
    domains = ["shopping", "shopping_admin", "reddit", "gitlab", "map"]
    for domain in domains:
        accs = per_domain_acc[domain]
        print(domain, sum(accs) / len(accs))
    overall_acc = [acc for accs in per_domain_acc.values() for acc in accs]
    print("Overall", sum(overall_acc) / len(overall_acc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_based", action="store_true")
    parser.add_argument("--result_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
