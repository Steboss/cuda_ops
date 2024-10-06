import os

if __name__ == "__main__":
    # Read inputs
    base_version = os.environ["SEMANTIC_VERSION"].split("-")[0]
    if os.getenv("DEV_VERSION_SUFFIX"):
        dev_version = f"{base_version}.dev0-{os.environ['DEV_VERSION_SUFFIX']}"
    else:
        dev_version = ""

    # Write outputs
    with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
        f.write(f"base-version={base_version}\n")
        f.write(f"dev-version={dev_version}\n")
