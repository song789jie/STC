import re
import torch
from torch.autograd import grad as torch_grad
from einops import rearrange
from torch.linalg import vector_norm
import os
import numpy as np


def exists(val):
    return val is not None


def read_tensor_from_txt(file_path, read_all=False, specified_number_of_lines=64):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    tensor_data = []

    if read_all:
        lines_to_read = lines[0: 64 - specified_number_of_lines]
    else:
        lines_to_read = lines[-specified_number_of_lines:]

    for line in lines_to_read:
        row = list(map(float, line.split()))
        tensor_data.append(row)

    tensor = torch.tensor(tensor_data)
    return tensor


def read_txt_files_to_numpy(directory, ratio):
    data_list = []
    for i, filename in enumerate(sort_files_by_number(os.listdir(directory))):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)

            read_all = (i == 0)

            tensor = read_tensor_from_txt(file_path, read_all=read_all,
                                          specified_number_of_lines=int(ratio * 64))

            numpy_array = tensor.numpy()
            data_list.append(numpy_array)

    combined_numpy_array = np.vstack(data_list)
    return combined_numpy_array


def delete_txt_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f'Deleted file: {file_path}')


def gradient_penalty(wave, output, weight=10):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs=output,
        inputs=wave,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim=1) - 1) ** 2).mean()


def sort_files_by_number(file_list):
    def extract_number(filename):
        match = re.search(r'_(\d+)\.txt$', filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')

    sorted_files = sorted(file_list, key=extract_number)
    return sorted_files
