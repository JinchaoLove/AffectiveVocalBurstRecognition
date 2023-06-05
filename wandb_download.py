# Reference: https://docs.wandb.ai/guides/track/public-api-guide
import wandb
api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    if '.pt' in file.name:
        file.download(replace=True)
