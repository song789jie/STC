def number_to_bits(O_number, B_number):
    B_bit_mapping = {
        0: '0',
        3: '10',
        4: '110',
        5: '1110',
        6: '11110'
    }

    O_bit_mapping = {
        0: '0',
        1/4: '10',
        1/2: '110',
        7/8: '1110',
    }

    if O_number not in O_bit_mapping:
        raise ValueError("Overlap_ratio must be one of the following: 0, 3, 4, 5, 6")

    if B_number not in B_bit_mapping:
        raise ValueError("Quantization_bit_width must be one of the following: 0, 3, 4, 5, 6")

    return O_bit_mapping[O_number], B_bit_mapping[B_number]


def bits_to_number(O_bits, B_bits):
    O_bit_mapping = {
        '0': 0,
        '10': 1/4,
        '110': 1/2,
        '1110': 7/8,
    }

    B_bit_mapping = {
        '0': 0,
        '10': 3,
        '110': 4,
        '1110': 5,
        '11110': 6,
    }

    if O_bits not in O_bit_mapping:
        raise ValueError("Overlap_ratio bits must be one of the following: '0', '10', '110', '1110'")

    if B_bits not in B_bit_mapping:
        raise ValueError("Quantization_bit_width bits must be one of the following: '0', '10', '110', '1110', '11110'")

    return O_bit_mapping[O_bits], B_bit_mapping[B_bits]
