import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from step1.DFT321 import smtdA
from step2.FSQ_NAE_coding import Coding_processing

demo_data_folder = './demo_data/'
demo_data_X_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_X.txt'
demo_data_Y_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_Y.txt'
demo_data_Z_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_Z.txt'

Length_frame = 64
Overlap_ratio = 0
Quantization_bit_width = 3
Use_residuals_quantization = False if Quantization_bit_width == 0 else True

FSQ_NAE = Coding_processing('Pre_train_weights/demo_weights.pth', 'codebook/huffman_codebook_fsq.npy')

if __name__ == "__main__":

    demo_data_X = np.loadtxt(os.path.join(demo_data_folder, demo_data_X_txt))
    demo_data_Y = np.loadtxt(os.path.join(demo_data_folder, demo_data_Y_txt))
    demo_data_Z = np.loadtxt(os.path.join(demo_data_folder, demo_data_Z_txt))
    # Transmitter #
    # Step1: Pre-Processing
    # Step1.1: Sliding Window
    Length_buffer = int(Length_frame * (1 - Overlap_ratio))
    num_windows = (len(demo_data_X) - Length_frame) // Length_buffer + 1

    total_frame_X = [demo_data_X[i:i + Length_frame] for i in range(0, num_windows * Length_buffer, Length_buffer)]
    total_frame_Y = [demo_data_Y[i:i + Length_frame] for i in range(0, num_windows * Length_buffer, Length_buffer)]
    total_frame_Z = [demo_data_Z[i:i + Length_frame] for i in range(0, num_windows * Length_buffer, Length_buffer)]
    # Step1.2: DFT321
    for i, (sub_frame_X, sub_frame_Y, sub_frame_Z) in \
            enumerate(zip(total_frame_X, total_frame_Y, total_frame_Z)):
        Ax, real_x, imag_x = smtdA(sub_frame_X)
        Ay, real_y, imag_y = smtdA(sub_frame_Y)
        Az, real_z, imag_z = smtdA(sub_frame_Z)
        # Sec. IV-A Eq. (1)
        As = np.sqrt(np.abs(Ax) ** 2 + np.abs(Ay) ** 2 + np.abs(Az) ** 2)
        real_sum = real_x + real_y + real_z
        imag_sum = imag_x + imag_y + imag_z
        theta = np.angle(real_sum + 1j * imag_sum)
        frame_x = np.fft.ifft(As * np.exp(1j * theta)).real
        # Step2: FSQ-NAE Model
        frame_x_orig = torch.from_numpy(frame_x).reshape((1, 1, 64)).float()
        # FSQ_NAE Encoder in FSQ_NAE.encode(·)
        FSQ_codewords, mean, std, frame_x_normalized = FSQ_NAE.encode(frame_x_orig)
        if Use_residuals_quantization:
            # FSQ_NAE Decoder in FSQ_NAE.decode(·)
            frame_xhat_normalized = FSQ_NAE.decode(FSQ_codewords, mean, std)

        # Step4: Bitstream Generation
        FSQ_codewords_bits, len_bits = FSQ_NAE.FSQ_codewords_huffman_encode(FSQ_codewords)

        print(FSQ_codewords, mean, std, frame_x_normalized)
        exit()

    print(num_windows)
    print(len())
    exit()
    # Plot Original Data
