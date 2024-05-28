

def mean_float_to_12bit(num):
    if num < -1 or num > 1:
        raise ValueError("Number out of range (-1, 1)")

    # Determine the sign bit (1 for negative, 0 for positive)
    sign_bit = 1 if num < 0 else 0

    # Take the absolute value for further processing
    abs_num = abs(num)

    # Determine the integer bit (1 if abs_num >= 0.5, else 0)
    int_bit = 1 if abs_num >= 0.5 else 0

    # Calculate the fractional part by removing the integer part
    fractional_part = abs_num - int_bit * 0.5

    # Scale the fractional part to fit into 10 bits
    scaled_fraction = int(fractional_part * 1024)

    # Convert to binary representation
    fractional_bits = format(scaled_fraction, '010b')

    # Combine the sign bit, integer bit, and fractional bits
    bit_representation = f'{sign_bit}{int_bit}{fractional_bits}'

    return bit_representation


def mean_bit_to_float(bit_str):
    if len(bit_str) != 12 or not all(c in '01' for c in bit_str):
        raise ValueError("Input must be a 12-bit binary string")

    # Extract sign, integer, and fractional bits
    sign_bit = int(bit_str[0])
    int_bit = int(bit_str[1])
    fractional_bits = bit_str[2:]

    # Convert fractional bits to a number
    fractional_part = int(fractional_bits, 2) / 1024.0

    # Calculate the decimal value
    num = int_bit * 0.5 + fractional_part

    # Apply the sign
    if sign_bit == 1:
        num = -num

    return num


def std_float_to_12bit(num):
    if num < -0.5 or num > 0.5:
        raise ValueError("Number out of range. Must be between -0.5 and 0.5")

    # Determine the sign bit
    if num < 0:
        sign_bit = '1'
        num = -num  # Make the number positive for further processing
    else:
        sign_bit = '0'

    # Scale the number to fit in 11 bits
    scaled_num = int(num * (2 ** 11))

    # Convert the scaled number to binary and pad to 11 bits
    fractional_bits = format(scaled_num, '011b')

    # Combine the sign bit and fractional bits
    binary_representation = sign_bit + fractional_bits

    return binary_representation


def std_bit_to_float(binary_code):
    if len(binary_code) != 12:
        raise ValueError("Binary code must be 12 bits long")

    # Extract the sign bit and fractional bits
    sign_bit = binary_code[0]
    fractional_bits = binary_code[1:]

    # Convert the fractional bits to an integer
    fractional_num = int(fractional_bits, 2)

    # Scale the number back to the original range
    num = fractional_num / (2**11)

    # Apply the sign
    if sign_bit == '1':
        num = -num

    return num

