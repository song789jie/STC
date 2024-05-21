import numpy as np

idxs_huffman_codec = [np.load('./codebook/counter_idxs_count_list_3.npy', allow_pickle=True).item(),
                      np.load('./codebook/counter_idxs_count_list_4.npy', allow_pickle=True).item(),
                      np.load('./codebook/counter_idxs_count_list_5.npy', allow_pickle=True).item(),
                      np.load('./codebook/counter_idxs_count_list_6.npy', allow_pickle=True).item()]


def indexes_huffman_encode(indices, bit):
    len_bits = 0
    string_bits = ''
    for i in range(len(indices)):
        indices_int = indices[i]
        bits = idxs_huffman_codec[int(bit) - 3].get(indices_int)
        if bits is None:
            print(indices_int)
        string_bits = string_bits + bits
        len_bits = len_bits + len(bits)
    return string_bits, len_bits


def indexes_huffman_decode(string_bits, bit):
    decoded_indices = []
    flipped_dict = {value: key for key, value in idxs_huffman_codec[int(bit) - 3].items()}
    index = 0
    current_code = ""
    for bit in string_bits:
        current_code += bit
        char = flipped_dict.get(current_code)
        if char is None:
            continue
        else:
            decoded_indices.extend([char])
            index = index + 1
            current_code = ""
    return decoded_indices
