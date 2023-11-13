import gdown

checkpoint_dir = "./checkpoints/"

gdown.download("https://drive.google.com/uc?id=1DPwnZa3iA-Fjn-SV0BLzW7KYvJywsd5L",
               checkpoint_dir + "final_run/" + "checkpoint.pth", quiet=True)
gdown.download("https://drive.google.com/uc?id=1tm3R9I8dWdwhMzDq3UfrVsNI7O1pY77n",
               checkpoint_dir + "final_run/" + "config.json", quiet=True)
