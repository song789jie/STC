import os
import numpy as np
import torch
from step1.DFT321 import smtdA
from step2.FSQ_NAE_model import Coding_processing
from step2.Mean_std_quant import float_to_12bit, bit_to_float
from step3.Residual_quantization import build_p, non_uniform_quant
from step3.Residual_indexes_huffman import indexes_huffman_encode, indexes_huffman_decode
from step4.Header import number_to_bits, bits_to_number


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

demo_data_folder = './demo_data/'
demo_data_X_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_X.txt'
demo_data_Y_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_Y.txt'
demo_data_Z_txt = 'Direct_-_1spike_Probe_-_polyesterPad_-_slow_Z.txt'

Length_frame = 64
Overlap_ratio = 0
Quantization_bit_width = 3
Use_residuals_quantization = False if Quantization_bit_width == 0 else True

FSQ_NAE = Coding_processing('Pre_train_weights/demo_weights.pth',
                            'codebook/huffman_codebook_fsq.npy')

if __name__ == "__main__":

    demo_data_X = np.loadtxt(os.path.join(demo_data_folder, demo_data_X_txt))
    demo_data_Y = np.loadtxt(os.path.join(demo_data_folder, demo_data_Y_txt))
    demo_data_Z = np.loadtxt(os.path.join(demo_data_folder, demo_data_Z_txt))
    # Transmitter Start#
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

            # Step3: Residual Quantization
            residual_frame = ((frame_x_normalized - frame_xhat_normalized)[0][0])
            # print(residual_frame)
            p_set = build_p(Bit_width=Quantization_bit_width - 1)

            residual_signal_hat, idxs = non_uniform_quant(residual_frame, p_set)
            residual_indexes = idxs.tolist()
            # print(residual_indexes)

            # Step4: Bitstream Generation - Residual_indexes
            Residual_indexes_bits, Residual_len_bits = (
                indexes_huffman_encode(residual_indexes, Quantization_bit_width))
        else:
            Residual_indexes_bits = ''
            Residual_len_bits = 0

        # Step4: Bitstream Generation - FSQ_codewords
        Mean_bit = float_to_12bit(mean)
        Std_bit = float_to_12bit(std)
        Overlap_ratio_bit, Quantization_bit_width_bit = (
            number_to_bits(Overlap_ratio, Quantization_bit_width))
        FSQ_codewords_bits, FSQ_len_bits = FSQ_NAE.FSQ_codewords_huffman_encode(FSQ_codewords)

        # print(Overlap_ratio_bit, Quantization_bit_width_bit, Mean_bit, Std_bit,
        #       FSQ_codewords_bits, Residual_indexes_bits)

        # print(len(Overlap_ratio_bit), len(Quantization_bit_width_bit),
        #       len(Mean_bit), len(Std_bit), len(FSQ_codewords_bits), len(Residual_indexes_bits))

        # Transmitter End#

        # Receiver Start #
        decode_Overlap_ratio, decode_Quantization_bit_width = (
            bits_to_number(Overlap_ratio_bit, Quantization_bit_width_bit))
        # print(decode_Overlap_ratio, decode_Quantization_bit_width)
        decode_mean = bit_to_float(Mean_bit)
        decode_std = bit_to_float(Std_bit)

        decode_FSQ_codewords = FSQ_NAE.FSQ_codewords_huffman_decode(FSQ_codewords_bits)
        if Use_residuals_quantization:
            decode_residual_indexes = indexes_huffman_decode(Residual_indexes_bits, decode_Quantization_bit_width)
            print(decode_residual_indexes)
        else:
            decode_residual_frame = 0

        exit()
        print(residual_indexes)


        # exit()

    print(num_windows)
    exit()
    # Plot Original Data
