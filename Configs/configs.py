import torch

run_in_colab = False
run_in_notebook = False
run_in_slurm = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
step = 0
num_images_per_dir = 1000
BASE_PATH = ''
# print(device)