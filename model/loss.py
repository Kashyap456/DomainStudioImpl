import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_Simple(nn.Module):
    def __init__(self):
        super(Loss_Simple, self).__init__()

    def forward(self, y_ada, y_pretrain):
        diff = y_ada - y_pretrain
        return torch.mean(diff * diff)


class Loss_PR(nn.Module):
    def __init__(self):
        super(Loss_PR, self).__init__()

    def forward(self, y_pred_sou, y_pred_ada):
        diff = y_pred_sou - y_pred_ada
        return torch.mean(diff * diff)


class PairwiseSimilarityLoss(nn.Module):
    def __init__(self, D):
        super(PairwiseSimilarityLoss, self).__init__()
        self.D = D
        self.sim = F.cosine_similarity
        self.sfm = F.softmax

    def forward(self, z_ada, z_sou):
        # Assume D represents a function that computes the denoised latent codes
        denoised_ada = self.D(z_ada).sample
        denoised_sou = self.D(z_sou).sample

        # Calculate similarity scores using broadcasting for all unique pairs
        # Expand dims to (batch_size, 1, features) for ada and (1, batch_size, features) for sou
        # to compute pairwise similarity
        # Reshape or flatten tensors to 2D (B, C*W*H)
        denoised_ada_flat = denoised_ada.reshape(denoised_ada.size(0), -1)
        denoised_sou_flat = denoised_sou.reshape(denoised_sou.size(0), -1)

        # Compute cosine similarity for all pairs (B, B)
        sim_matrix_ada = self.sim(denoised_ada_flat.unsqueeze(
            1), denoised_ada_flat.unsqueeze(0), dim=2)
        sim_matrix_sou = self.sim(denoised_sou_flat.unsqueeze(
            1), denoised_sou_flat.unsqueeze(0), dim=2)

        # Mask out the self-similarity (diagonal elements of the similarity matrix)
        mask = torch.eye(sim_matrix_ada.size(
            0)).bool().to(z_ada.device)
        sim_matrix_ada = sim_matrix_ada.masked_fill(mask, float('-inf'))
        sim_matrix_sou = sim_matrix_sou.masked_fill(mask, float('-inf'))

        # Apply softmax to the non-diagonal elements to get the probabilities
        p_ada = self.sfm(sim_matrix_ada, dim=1)
        p_sou = self.sfm(sim_matrix_sou, dim=1)

        epsilon = 1e-8
        p_ada = p_ada + epsilon
        p_sou = p_sou + epsilon

        # Calculate KL divergence
        kl_divergence = F.kl_div(p_ada.log(), p_sou, reduction='batchmean')

        assert not torch.isnan(denoised_ada).any(), "NaNs in denoised_ada"
        assert not torch.isnan(denoised_sou).any(), "NaNs in denoised_sou"
        assert not torch.isnan(p_ada).any(), "NaNs in p_ada"
        assert not torch.isnan(p_sou).any(), "NaNs in p_sou"
        assert not torch.isnan(kl_divergence).any(), "NaNs in KL divergence"

        return kl_divergence


class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the 2D filters based on the outer product of the 1D filters
        lh_base = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2.0
        hl_base = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2.0
        hh_base = torch.tensor([[-1.0, 1.0], [1.0, -1.0]]) / 2.0

        # Initialize filters for 3 input channels and 3 output channels
        self.register_buffer('lh_filter', torch.stack(
            [lh_base] * 3).unsqueeze(1))
        self.register_buffer('hl_filter', torch.stack(
            [hl_base] * 3).unsqueeze(1))
        self.register_buffer('hh_filter', torch.stack(
            [hh_base] * 3).unsqueeze(1))

    def forward(self, x):
        # Ensure input x has a batch and channel dimension
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add a channel dimension if it's not present

        # Apply filters to input x independently for each channel
        lh = F.conv2d(x, self.lh_filter, stride=2, padding=0, groups=3)
        hl = F.conv2d(x, self.hl_filter, stride=2, padding=0, groups=3)
        hh = F.conv2d(x, self.hh_filter, stride=2, padding=0, groups=3)

        # Normalize and combine the LH, HL, HH components
        # Note: Direct normalization to 0-255 might not be ideal for training purposes.
        # Here we normalize them to have a mean of 0 and a std of 1 for combining
        def normalize(tensor):
            mean = tensor.mean([2, 3], keepdim=True)
            std = tensor.std([2, 3], keepdim=True)
            return (tensor - mean) / (std + 1e-5)

        lh_norm = normalize(lh)
        hl_norm = normalize(hl)
        hh_norm = normalize(hh)

        # Since we're working in a forward function for training, we combine them by averaging
        # This is different from the visualization approach and ensures the result remains differentiable
        high_freq_representation = (lh_norm + hl_norm + hh_norm) / 3

        # Note: This operation reduces the channel dimension to the average of the high-frequency components
        # If you intended to keep the channel dimensionality, consider modifying this approach
        return high_freq_representation


class Loss_HF(nn.Module):
    def __init__(self, D):
        super(Loss_HF, self).__init__()
        self.hwt = HaarWaveletTransform()
        self.sim = F.cosine_similarity
        self.sfm = F.softmax
        self.D = D

    def forward(self, z_ada, z_sou):
        # Assume D represents a function that computes the denoised latent codes
        # use the transform to extract the high frequency (fine-grained) details
        denoised_ada = self.hwt(self.D(z_ada).sample)
        denoised_sou = self.hwt(self.D(z_sou).sample)

        # Calculate similarity scores using broadcasting for all unique pairs
        # Expand dims to (batch_size, 1, features) for ada and (1, batch_size, features) for sou
        # to compute pairwise similarity
        denoised_ada_flat = denoised_ada.reshape(denoised_ada.size(0), -1)
        denoised_sou_flat = denoised_sou.reshape(denoised_sou.size(0), -1)

        # Compute cosine similarity for all pairs (B, B)
        sim_matrix_ada = self.sim(denoised_ada_flat.unsqueeze(
            1), denoised_ada_flat.unsqueeze(0), dim=2)
        sim_matrix_sou = self.sim(denoised_sou_flat.unsqueeze(
            1), denoised_sou_flat.unsqueeze(0), dim=2)

        # Mask out the self-similarity (diagonal elements of the similarity matrix)
        mask = torch.eye(sim_matrix_ada.size(
            0)).bool().to(z_ada.device)
        sim_matrix_ada = sim_matrix_ada.masked_fill(mask, float('-inf'))
        sim_matrix_sou = sim_matrix_sou.masked_fill(mask, float('-inf'))

        # Apply softmax to the non-diagonal elements to get the probabilities
        p_ada = self.sfm(sim_matrix_ada, dim=1)
        p_sou = self.sfm(sim_matrix_sou, dim=1)

        epsilon = 1e-8
        p_ada = p_ada + epsilon
        p_sou = p_sou + epsilon

        # Calculate KL divergence
        kl_divergence = F.kl_div(p_ada.log(), p_sou, reduction='batchmean')

        assert not torch.isnan(denoised_ada).any(), "NaNs in denoised_ada"
        assert not torch.isnan(denoised_sou).any(), "NaNs in denoised_sou"
        assert not torch.isnan(p_ada).any(), "NaNs in p_ada"
        assert not torch.isnan(p_sou).any(), "NaNs in p_sou"
        assert not torch.isnan(kl_divergence).any(), "NaNs in KL divergence"

        return kl_divergence


class Loss_HFMSE(nn.Module):
    def __init__(self, D):
        super(Loss_HFMSE, self).__init__()
        self.hwt = HaarWaveletTransform()
        self.D = D

    def forward(self, z_ada, x_init):
        diff = self.hwt(self.D(z_ada).sample) - self.hwt(x_init)
        return torch.mean(diff * diff)


class DomainLoss(nn.Module):
    def __init__(self, D):
        super(DomainLoss, self).__init__()
        # Loss Components
        self.l_simp = Loss_Simple()
        self.l_pr = Loss_PR()
        self.l_img = PairwiseSimilarityLoss(D)
        self.l_hf = Loss_HF(D)
        self.l_hfmse = Loss_HFMSE(D)

        # Loss Weights
        self.l1 = 1
        self.l2 = 250
        self.l3 = 250
        self.l4 = 0.6

    def forward(self, z_ada, z, z_pr_sou, z_pr_ada, x_init):
        # Compute each loss component
        loss_simp = self.l_simp(z_ada, z)
        loss_pr = self.l1 * self.l_pr(z_pr_sou, z_pr_ada)
        loss_img = self.l2 * self.l_img(z_ada, z_pr_ada)
        loss_hf = self.l3 * self.l_hf(z_ada, z_pr_ada)
        loss_hfmse = self.l4 * self.l_hfmse(z_ada, x_init)

        # Log or print each component
        print(f"Loss Simple: {loss_simp.item()}")
        print(f"Loss PR: {loss_pr.item()}")
        print(f"Loss IMG: {loss_img.item()}")
        print(f"Loss HF: {loss_hf.item()}")
        print(f"Loss HFMSE: {loss_hfmse.item()}")

        # Return the total loss
        total_loss = loss_simp + loss_pr + loss_img + loss_hf + loss_hfmse
        return total_loss
