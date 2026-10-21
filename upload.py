from huggingface_hub import HfApi

api = HfApi()
username = api.whoami()["name"]
repo_id = f"{username}/distilbert-sst2-int8-affine-per-channel"

api.create_repo(repo_id, private=True, exist_ok=True)
api.upload_folder(
    folder_path="hf_release",
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["__pycache__/*", "*.pyc"],
)
print("Done: https://huggingface.co/" + repo_id)