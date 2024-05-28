import torch


def build_p(Bit_width=2):
    p_0 = [0.]
    p_1 = [0.]
    p_2 = [0.]

    if Bit_width == 2:
        for i in range(3):
            p_0.append(2 ** (-i - 1))
    elif Bit_width == 4:
        for i in range(3):
            p_0.append(2 ** (-2 * i - 1))
            p_1.append(2 ** (-2 * i - 2))
    elif Bit_width == 6:
        for i in range(3):
            p_0.append(2 ** (-3 * i - 1))
            p_1.append(2 ** (-3 * i - 2))
            p_2.append(2 ** (-3 * i - 3))
    elif Bit_width == 3:
        for i in range(3):
            if i < 2:
                p_0.append(2 ** (-i - 1))
            else:
                p_1.append(2 ** (-i - 1))
                p_0.append(2 ** (-i - 2))
    elif Bit_width == 5:
        for i in range(3):
            if i < 2:
                p_0.append(2 ** (-2 * i - 1))
                p_1.append(2 ** (-2 * i - 2))
            else:
                p_2.append(2 ** (-2 * i - 1))
                p_0.append(2 ** (-2 * i - 2))
                p_1.append(2 ** (-2 * i - 3))

    R = []
    p_0.sort()
    p_1.sort()
    p_2.sort()
    for a in p_0:
        for b in p_1:
            for c in p_2:
                R.append((a + b + c))

    R = torch.Tensor(list(R))
    R = R.mul(1.0 / torch.max(R))
    return R


def non_uniform_quant(tensor, p_set):
    def p_quant(x, value_s):
        shape = x.shape
        x_hat = x.view(-1)
        sign = x.sign()
        value_s = value_s.type_as(x)
        x_hat = x_hat.abs()
        idexes = (x_hat.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        x_hat = value_s[idexes].view(shape).mul(sign)
        x_hat = x_hat
        x_hat = (x_hat - x).detach() + x
        return x_hat, idexes

    data = tensor / 2
    data = data.clamp(-1, 1)
    data_q, idxs = p_quant(data, p_set)
    data_q = data_q * 2

    return data_q, idxs


def non_uniform_dequant(idxs, p_set):
    def p_dequant(idxs, value_s):
        idxs = torch.tensor(idxs)
        shape = idxs.shape
        x_hat = value_s[idxs].view(shape)
        return x_hat

    data_q = p_dequant(idxs, p_set)
    data_q = data_q * 2.0

    return data_q




