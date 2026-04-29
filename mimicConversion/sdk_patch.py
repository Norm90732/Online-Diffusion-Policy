import os
import lightwheel_sdk.loader as loader #pyrefly:ignore 


if not hasattr(loader, "ENDPOINT"):
    loader.ENDPOINT = os.environ.get("LW_API_ENDPOINT", "https://api-dev.lightwheel.net")
