import json
from pathlib import Path

def gen_metrics_file_path(base_dir: str, secondary_prefix: str, tertiary_prefix: str):
    base_path = Path(base_dir)

    def file_finder(path: Path, prefix):
        for child in path.iterdir():
            if child.name.startswith(prefix):
                yield child

    for secondary in file_finder(base_path, secondary_prefix):
        if secondary.is_dir():
            # print(f"secondary: {secondary}")
            for teriary in file_finder(secondary, tertiary_prefix):
                # print(f"teriary: {teriary}")
                try:
                    yield next(file_finder(teriary, "val_metrics.json"))
                except:
                    print(f"can't find val_metrics in {teriary}")

if __name__=="__main__":
    # BASE_DIR, SECOND, TERTIARY, metric = sys.argv[1:]

    BASE_DIR = "___el_ep_2022"
    SECOND = "ep"
    TERTIARY = "student"
    metric = "NDCG@10"

    best_result = 0.
    best_path = "error"

    for metrics_path in gen_metrics_file_path(BASE_DIR, SECOND, TERTIARY):
        with metrics_path.open('r') as f:
            j = json.load(f)
            
            if j[metric] > best_result:
                best_result = j[metric]
                best_path = Path(metrics_path)
    
    if not isinstance(best_path, str):
        print(best_path.parent.joinpath("checkpoint/best_acc_model.pth"))
    else:
        raise FileNotFoundError(f"best path {best_path} not found.")
