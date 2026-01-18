import wandb

# Direct API key login (put your key in quotes!)
wandb.login(key='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')

# Initialize W&B project
wandb.init(project="isl-sign-language-translator")

print("W&B Setup Complete!")
print(f"Dashboard: {wandb.run.get_url()}")

# Log a test metric
wandb.log({"test_metric": 1.0, "status": "connected"})

print("Test metric logged successfully!")

# Finish the run
wandb.finish()

print("\n W&B is ready to use in your ISL project!")