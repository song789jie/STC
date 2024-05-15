import torch
import torch.nn.functional as F


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def SSIM_1D_loss(fake, real):

    c1 = c2 = c3 = 0.005
    mu_real_p = real.mean(dim=-1)
    mu_fake_p = fake.mean(dim=-1)

    l_real_fake = (2 * mu_fake_p * mu_real_p + c1) / (torch.pow(mu_fake_p, 2) + torch.pow(mu_real_p, 2) + c1)
    # print(l_real_fake)
    deta_real_p = torch.std(real, dim=-1)
    deta_fake_p = torch.std(fake, dim=-1)

    c_real_fake = (2 * deta_real_p * deta_fake_p + c2) / (torch.pow(deta_real_p, 2) + torch.pow(deta_fake_p, 2) + c2)
    # print(c_real_fake)

    s_real_fake = ((torch.sum((real - mu_real_p) * (fake - mu_fake_p), dim=-1) + c3) /
                   (torch.sqrt(torch.sum((real - mu_real_p) ** 2, dim=-1)) * torch.sqrt(
                       torch.sum((fake - mu_fake_p) ** 2, dim=-1)) + c3))

    sp = torch.pow(l_real_fake, 1) * torch.pow(c_real_fake, 1) * torch.pow(s_real_fake, 1)
    return 1 - sp.mean()


if __name__ == '__main__':
    data1 = torch.randn(size=(42, 1, 64))
    data2 = torch.randn(size=(42, 1, 64))
    print(SSIM_1D_loss(data1, data2))