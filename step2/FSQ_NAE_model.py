import torch
import numpy as np
import intel_npu_acceleration_library
from model.FSQ_NAE import FSQ_NAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Coding_processing:
    def __init__(self,
                 model_path,
                 huffman_codebook_path,
                 ):
        super(Coding_processing, self).__init__()

        self.load_model = torch.load(model_path, map_location=device)
        self.model_total = FSQ_NAE()
        self.model_total.load_state_dict(self.load_model)
        # for NPU #
        # self.model_total = intel_npu_acceleration_library.compile(self.model_total, dtype=torch.float16)

        self.model_total.to(device)
        self.model_total.eval()
        self.huffman_codec = np.load(huffman_codebook_path, allow_pickle=True).item()
        self.indices_shape = [1, 6]

    def encode(self, data_orig):

        x = data_orig.to(device)
        # Normalization
        mean_raw_x = torch.mean(x, dim=-1, keepdim=True).detach()
        x = x - mean_raw_x
        std_raw_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_raw_x
        x_stationary = x

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
        decoded_indices = torch.zeros(6).type(torch.int32).to(device)
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



