import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils import read_txt_files_to_numpy, delete_txt_files
from step1.DFT321 import smtdA
from step2.FSQ_NAE_model import Coding_processing
from step2.Mean_std_quant import mean_float_to_12bit, mean_bit_to_float, std_float_to_12bit, std_bit_to_float
from step3.Residual_quantization import build_p, non_uniform_quant, non_uniform_dequant
from step3.Residual_indexes_huffman import indexes_huffman_encode, indexes_huffman_decode
from step4.Header import number_to_bits, bits_to_number
from vibromaf.metrics.snr import snr, psnr
from vibromaf.metrics.stsim import st_sim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

demo_data_folder = './demo_data/'
demo_data_orig_save_folder = './demo_save_data/'
demo_data_recon_save_folder = './demo_save_recon_data'
delete_txt_files(demo_data_orig_save_folder)
delete_txt_files(demo_data_recon_save_folder)
demo_data_X_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_X.txt'
demo_data_Y_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_Y.txt'
demo_data_Z_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_Z.txt'

Length_frame = 64
Overlap_ratio = 0
Quantization_bit_width = 0
Use_residuals_quantization = False if Quantization_bit_width == 0 else True

FSQ_NAE = Coding_processing('Pre_train_weights/demo_weights.pth',
                            'codebook/huffman_codebook_fsq.npy')

if __name__ == "__main__":

    demo_data_X = np.loadtxt(os.path.join(demo_data_folder, demo_data_X_txt))
    demo_data_Y = np.loadtxt(os.path.join(demo_data_folder, demo_data_Y_txt))
    demo_data_Z = np.loadtxt(os.path.join(demo_data_folder, demo_data_Z_txt))
    # Transmitter Start #
    # Step1: Pre-Processing
    # Step1.1: Sliding Window
    Length_buffer = int(Length_frame * (1 - Overlap_ratio))
    num_windows = (len(demo_data_X) - Length_frame) // Length_buffer + 1

    total_frame_X = [demo_data_X[i:i + Length_frame] for i in range(0, num_windows * Length_buffer, Length_buffer)]
    total_frame_Y = [demo_data_Y[i:i + Length_frame] for i in range(0, num_windows * Length_buffer, Length_buffer)]
    total_frame_Z = [demo_data_Z[i:i + Length_frame] for i in range(0, num_windows * Length_buffer, Length_buffer)]
    # Step1.2: DFT321
    total_bits = 0
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
        numpy_frame_x_orig = np.asarray(frame_x_orig).reshape(64)
        np.savetxt(os.path.join(demo_data_orig_save_folder, demo_data_X_txt[:-5] + str(i) + '.txt'), numpy_frame_x_orig)

        # FSQ_NAE Encoder in FSQ_NAE.encode(·)
        FSQ_codewords, mean, std, frame_x_normalized = FSQ_NAE.encode(frame_x_orig)
        residual_signal_hat = 0
        if Use_residuals_quantization:
            # FSQ_NAE Decoder in FSQ_NAE.decode(·)
            frame_xhat_normalized = FSQ_NAE.decode(FSQ_codewords, mean, std)

            # Step3: Residual Quantization
            residual_frame = ((frame_x_normalized - frame_xhat_normalized)[0][0])
            # print(residual_frame)
            p_set = build_p(Bit_width=Quantization_bit_width - 1)

            residual_signal_hat, idxs = non_uniform_quant(residual_frame, p_set)
            # print(residual_frame, residual_signal_hat)
            residual_indexes = idxs.tolist()
            # print(residual_indexes)

            # Step4: Bitstream Generation - Residual_indexes
            Residual_indexes_bits, Residual_len_bits = (
                indexes_huffman_encode(residual_indexes, Quantization_bit_width))
        else:
            Residual_indexes_bits = ''
            Residual_len_bits = 0

        # Step4: Bitstream Generation - FSQ_codewords
        Mean_bit = mean_float_to_12bit(mean)
        Std_bit = std_float_to_12bit(std)

        Overlap_ratio_bit, Quantization_bit_width_bit = (
            number_to_bits(Overlap_ratio, Quantization_bit_width))
        FSQ_codewords_bits, FSQ_len_bits = FSQ_NAE.FSQ_codewords_huffman_encode(FSQ_codewords)

        # Transmitter End #
        total_bits = (total_bits + len(Overlap_ratio_bit) + len(Quantization_bit_width_bit)
                      + len(Mean_bit) + len(Std_bit) + FSQ_len_bits + Residual_len_bits)

        # Lossless Communication #

        # Receiver Start #
        decode_Overlap_ratio, decode_Quantization_bit_width = (
            bits_to_number(Overlap_ratio_bit, Quantization_bit_width_bit))

        # print('Decode Overlap Ratio vs Overlap Ratio:', decode_Overlap_ratio, Overlap_ratio)
        decode_mean = torch.tensor(mean_bit_to_float(Mean_bit)).view(mean.shape)
        decode_std = torch.tensor(std_bit_to_float(Std_bit)).view(std.shape)
        # print('Decode Mean vs Mean:', decode_mean, mean)
        # print('Decode Std vs Std:', decode_std, std)
        decode_FSQ_codewords = FSQ_NAE.FSQ_codewords_huffman_decode(FSQ_codewords_bits)
        if Use_residuals_quantization:
            decode_residual_indexes = indexes_huffman_decode(Residual_indexes_bits, decode_Quantization_bit_width)
            p_set = build_p(Bit_width=Quantization_bit_width - 1)
            decode_residual_signal_hat = non_uniform_dequant(decode_residual_indexes, p_set)
            decode_residual_frame = residual_signal_hat
            # print('Decode Residual vs Residual:', decode_residual_signal_hat, residual_signal_hat)
        else:
            decode_residual_frame = 0

        decode_frame_xhat_normalized = FSQ_NAE.decode(decode_FSQ_codewords, decode_mean, decode_std)
        # print('Decode normalized xhat vs normalized xhat:', decode_frame_xhat_normalized, frame_x_normalized)
        decode_frame_xhat = ((decode_frame_xhat_normalized + decode_residual_frame) * decode_std) + decode_mean
        numpy_recon_x_orig = np.asarray(decode_frame_xhat.detach().numpy()).reshape(64)
        np.savetxt(os.path.join(demo_data_recon_save_folder, demo_data_X_txt[:-5] + str(i) + '.txt'), numpy_recon_x_orig)

    total_orig = read_txt_files_to_numpy(demo_data_orig_save_folder, ratio=Overlap_ratio)
    total_recon = read_txt_files_to_numpy(demo_data_recon_save_folder, ratio=Overlap_ratio)

    print('bit rate:', total_bits / total_orig.shape[0] * 2800 / 1024, 'kbps')
    print('compression ratio:', 16 * total_orig.shape[0] / total_bits)
    print('SNR:', snr(total_recon, total_orig))
    print('PSNR:', psnr(total_recon, total_orig))
    print('ST-SIM:', st_sim(total_recon, total_orig))

    x = np.arange(1, len(total_orig) + 1)
    plt.figure()
    plt.plot(x, total_orig, label='orig')
    plt.plot(x, total_recon, label='recon')
    plt.legend()
    plt.show()

