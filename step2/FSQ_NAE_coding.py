import torch
import numpy as np
import struct
from vector_quantize_pytorch import FSQ

from model.FSQ_NAE import FSQ_NAE

result_list = ['{0:08b}'.format(i & 0xff) for i in range(-128, 128)]
result_list = [item.replace('0', '0').replace('1', '1') for item in result_list]


def mean_std_dequantization(mean_bit_string, std_bit_string):
    mean_bytes_list = [mean_bit_string[i:i + 8] for i in range(0, len(mean_bit_string), 8)]
    mean_byte_data = [int(byte, 2) for byte in mean_bytes_list]
    mean_byte_data = bytes(mean_byte_data)
    mean = torch.tensor(struct.unpack('e', mean_byte_data)[0]).to('cuda')

    std_bytes_list = [std_bit_string[i:i + 8] for i in range(0, len(std_bit_string), 8)]
    std_byte_data = [int(byte, 2) for byte in std_bytes_list]
    std_byte_data = bytes(std_byte_data)
    std = torch.tensor(struct.unpack('e', std_byte_data)[0]).to('cuda')

    mean_return = mean.reshape(1, 1, 1).float()
    std_return = std.reshape(1, 1, 1).float()

    return mean_return, std_return


def mean_std_quantization(mean, std):
    mean_bit_representation = struct.pack('e', mean.item())
    # print('mean', mean_bit_representation)
    mean_bit_string = ''.join(f'{byte:08b}' for byte in mean_bit_representation)
    # print(mean_bit_string)

    std_bit_representation = struct.pack('e', std.item())
    std_bit_string = ''.join(f'{byte:08b}' for byte in std_bit_representation)
    return mean_bit_string, std_bit_string


class Coding_processing:
    def __init__(self,
                 model_path,
                 huffman_codebook_path,
                 ):
        super(Coding_processing, self).__init__()

        self.load_model = torch.load(model_path)
        self.model_total = FSQ_NAE()
        self.model_total.load_state_dict(self.load_model)
        self.model_total.to('cuda')
        self.model_total.eval()
        self.huffman_codec = np.load(huffman_codebook_path, allow_pickle=True).item()

    def encode(self, data_orig):

        x = data_orig.to('cuda')
        # Normalization
        mean_raw_x = torch.mean(x, dim=-1, keepdim=True).detach()
        x = x - mean_raw_x
        std_raw_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_raw_x
        x_stationary = x.deatch()

        # Non-stationary Attention: Tau Delta (Eq.6)
        tau = self.model_total.tau_learner(x, std_raw_x).exp()
        delta = self.model_total.delta_learner(x, mean_raw_x)

        x_embedding = self.model_total.embedding(x)
        x_embedding = x_embedding.transpose(1, 2)
        # Non-stationary Attention: Network in self.model_total.encoder(·)
        x, attn_encoder = self.model_total.encoder(x_embedding, tau, delta)

        # Convolution Layers
        x = x.transpose(1, 2)
        x = self.model_total.encoder_causal_conv(x)

        # FSQ: Representation to Codewords
        xhat, codewords = self.model_total.fsq(x)

        return codewords, mean_raw_x, std_raw_x, x_stationary

    def decode(self, codewords, mean, std):

        # FSQ: Codewords to Representation
        xhat = self.model_total.fsq.indices_to_codes(codewords)
        # Transpose Convolution Layers
        xhat = self.model_total.decoder_causal_conv(xhat)
        # Assumed Non-stationary Attention: Tau Delta
        xhat = xhat.transpose(1, 2)
        xhat, attn_decoder = self.model_total.decoder(xhat, std ** 2, mean)
        xhat = xhat.transpose(1, 2)

        return xhat

    def FSQ_codewords_huffman_encode(self, indices):
        len_bits = 0
        string_bits = ''
        for i in range(len(indices[0])):
            indices_int = indices[0][i].item()
            bits = self.huffman_codec.get(indices_int)
            if bits is None:
                print(indices_int)
            string_bits = string_bits + bits
            len_bits = len_bits + len(bits)
        return string_bits, len_bits

    def FSQ_codewords_huffman_decode(self, string_bits):
        decoded_indices = torch.zeros(self.indices_shape).type(torch.int32).to('cuda')
        flipped_dict = {value: key for key, value in self.huffman_codec.items()}
        index = 0
        current_code = ""
        for bit in string_bits:
            current_code += bit
            char = flipped_dict.get(current_code)
            if char is None:
                continue
            else:
                decoded_indices[index] = char
                index = index + 1
                current_code = ""
        decoded_indices = torch.unsqueeze(decoded_indices, dim=0)
        return decoded_indices



