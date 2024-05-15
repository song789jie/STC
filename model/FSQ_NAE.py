import torch
from torch import nn
from vector_quantize_pytorch import FSQ
from functools import partial
import torch.nn.functional as F

from model.custom_module import MultiScaleDiscriminator, MLP_with_stats, DataEmbedding, \
    DSAttention_Layer, DSAttention, CausalConv1d, EncoderBlock, DecoderBlock, Attention_Layer, \
    Attention
from model.custom_loss import hinge_discr_loss, hinge_gen_loss, SSIM_1D_loss
from src.utils import exists, gradient_penalty


class FSQ_NAE(nn.Module):
    def __init__(
            self,
            *,
            FSQ_levels=[8, 5, 5, 5],
            FSQ_codebook_dim=6,
            channels=16,
            strides=(2, 2, 2, 2),
            channel_mults=(2, 2, 4, 4),
            pad_mode='constant',
            input_channels=1,
            embedding_dim=16,
            discr_multi_scales=(1, 0.5, 0.25),
            mse_loss_weight=1.,
            adversarial_loss_weight=1.,
            feature_loss_weight=10.,
            SSIM_loss_weight=0.1
    ):
        super().__init__()

        self.mse_loss_weight = mse_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.SSIM_loss_weight = SSIM_loss_weight

        self.single_channel = input_channels == 1
        self.strides = strides
        self.embedding_dim = embedding_dim

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))
        # print(chan_in_out_pairs)
        self.tau_learner = MLP_with_stats(output_dim=1)
        self.delta_learner = MLP_with_stats(output_dim=64)

        self.discr_multi_scales = discr_multi_scales
        self.discriminators = nn.ModuleList([MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))])

        discr_rel_factors = [int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])]
        self.downsamples = nn.ModuleList(
            [nn.Identity()] + [nn.AvgPool1d(2 * factor, stride=factor, padding=factor) for factor in discr_rel_factors])

        self.embedding = DataEmbedding(c_in=1, d_model=self.embedding_dim)
        #
        self.encoder = DSAttention_Layer(DSAttention(), d_model=self.embedding_dim)
        self.decoder = Attention_Layer(Attention(), d_model=self.embedding_dim)
        encoder_causal_conv_blocks = []
        decoder_causal_conv_blocks = []
        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_causal_conv_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride,
                                                           False, pad_mode))

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_causal_conv_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride,
                                                           False, pad_mode))

        self.encoder_causal_conv = nn.Sequential(
            CausalConv1d(self.embedding_dim, channels, 7, pad_mode=pad_mode),
            *encoder_causal_conv_blocks,
            CausalConv1d(layer_channels[-1], FSQ_codebook_dim, 3, pad_mode=pad_mode)
        )

        self.decoder_causal_conv = nn.Sequential(
            CausalConv1d(FSQ_codebook_dim, layer_channels[-1], 3, pad_mode=pad_mode),
            *decoder_causal_conv_blocks,
            CausalConv1d(channels, embedding_dim, 7, pad_mode=pad_mode)
        )

        self.fsq = FSQ(FSQ_levels)

    def non_discr_parameters(self):
        return [
            # 用*返回具体的值，而不是地址
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.embedding.parameters(),
            *self.encoder_causal_conv.parameters(),
            *self.decoder_causal_conv.parameters(),
            *(self.tau_learner.parameters() if exists(self.tau_learner) else []),
            *(self.delta_learner.parameters() if exists(self.delta_learner) else []),
        ]

    def forward(
            self,
            x,
            return_encoded=False,
            return_recons_only=False,
            return_discr_loss=False,
            return_discr_losses_separately=False,
            return_loss_breakdown=False,
            apply_grad_penalty=False,
    ):
        raw_x = x.clone().detach()
        mean_raw_x = torch.detach(torch.mean(x, dim=-1, keepdim=True))
        x = x - mean_raw_x
        std_raw_x = torch.detach(torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5))
        x = x / std_raw_x

        orig_stationary_x = x.clone()
        tau = self.tau_learner(raw_x, std_raw_x).exp()
        delta = self.delta_learner(raw_x, mean_raw_x)
        x_embedding = self.embedding(x)
        x_embedding = x_embedding.transpose(1, 2)
        x, attn_encoder = self.encoder(x_embedding, tau, delta)
        x = x.transpose(1, 2)
        x = self.encoder_causal_conv(x)

        xhat, indices = self.fsq(x)
        if return_encoded:
            return xhat, indices

        xhat = self.decoder_causal_conv(xhat)
        xhat = xhat.transpose(1, 2)
        xhat, attn_decoder = self.decoder(xhat, std_raw_x ** 2, mean_raw_x)
        xhat = xhat.transpose(1, 2)
        recon_stationary_x = xhat.clone()

        x_recon = xhat * std_raw_x + mean_raw_x
        if return_recons_only:
            return x_recon

        if return_discr_loss:
            real, fake = orig_stationary_x, recon_stationary_x.detach()
            discr_losses = []
            discr_grad_penalties = []
            scaled_real, scaled_fake = real, fake

            for discr, downsample in zip(self.discriminators, self.downsamples):

                scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

                real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake))
                one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

                discr_losses.append(one_discr_loss)
                if apply_grad_penalty:
                    discr_grad_penalties.append(gradient_penalty(scaled_real, one_discr_loss))

            if not return_discr_losses_separately:
                all_discr_losses = torch.stack(discr_losses).mean()
                return all_discr_losses

            discr_losses_pkg = []

            discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in
                                     zip(self.discr_multi_scales, discr_losses)])

            discr_losses_pkg.extend(
                [(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in
                 zip(self.discr_multi_scales, discr_grad_penalties)])
            # print(discr_losses_pkg)
            return discr_losses_pkg

        mse_loss = F.mse_loss(orig_stationary_x, recon_stationary_x)
        SSIM_loss = SSIM_1D_loss(recon_stationary_x, orig_stationary_x)
        adversarial_losses = []

        discr_intermediates = []

        scaled_real, scaled_fake = orig_stationary_x, recon_stationary_x

        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

            (real_logits, real_intermediates), (fake_logits, fake_intermediates) = map(
                partial(discr, return_intermediates=True), (scaled_real, scaled_fake))

            discr_intermediates.append((real_intermediates, fake_intermediates))

            one_adversarial_loss = hinge_gen_loss(fake_logits)
            adversarial_losses.append(one_adversarial_loss)

        feature_losses = []

        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(real_intermediate, fake_intermediate) for real_intermediate, fake_intermediate
                      in zip(real_intermediates, fake_intermediates)]
            feature_losses.extend(losses)

        feature_loss = torch.stack(feature_losses).mean()
        adversarial_loss = torch.stack(adversarial_losses).mean()

        total_loss = (mse_loss * self.mse_loss_weight) \
                     + (adversarial_loss * self.adversarial_loss_weight) \
                     + (feature_loss * self.feature_loss_weight) \
                     + (SSIM_loss * self.SSIM_loss_weight)

        if return_loss_breakdown:
            return total_loss, (mse_loss, adversarial_loss, feature_loss, SSIM_loss)

        return total_loss


if __name__ == '__main__':
    pass

