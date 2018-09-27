import os

# Training Parameters
os.environ["train"] = "False"
os.environ["batch_size"] = "128"
os.environ["num_epochs"] = "6"
os.environ["display_step"] = "32"

# Network Parameters
os.environ["input_size_h"] = "192"
os.environ["input_size_w"] = "256"
os.environ["num_channels"] = "4"
