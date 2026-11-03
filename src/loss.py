# todo: implement custom loss function

# auxiliary losses ideas:
# - non-binary densities (encourage densities to be close to 0 or 1)
# - floating material


import torch
import math
import warnings

# ! todo: make patch_size a parameter or (better) pass pred as 2D tensor directly

def _density_index_to_absolute_coordinates(idx):
    return (idx // 64, idx % 64)

def _absolute_coordinates_to_density_index(coords):
    return coords[0] * 64 + coords[1]

def _load_adjacent_elements(load):
     #! ai aided code snippet: helped me with all the indexing and transformation logic for adjacent elements
    """Row/col indices of the up to 4 elements touching the load node, plus a validity mask."""
    node_cols, node_rows = (load[:, :2] * 64).round().long().unbind(dim=1)
    if node_cols.min() < 0 or node_cols.max() > 64 or node_rows.min() < 0 or node_rows.max() > 64:
        raise ValueError("Load coordinates are out of bounds.")
    rows = torch.stack((node_rows - 1, node_rows - 1, node_rows, node_rows), dim=1)
    cols = torch.stack((node_cols - 1, node_cols, node_cols - 1, node_cols), dim=1)
    valid = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
    return rows.clamp(0, 63), cols.clamp(0, 63), valid

class VolumeFractionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_volume_fraction):
        """Penalizes the difference in volume fraction between the predicted and target designs."""
        
        design_domain_volume = torch.numel(pred[0]) # should be 64*64
        predicted_volume_fraction = torch.sum(pred, dim=1) / design_domain_volume
        loss = (target_volume_fraction-predicted_volume_fraction).abs()
        return loss.mean() # average over batch


class LoadDiscrepancyLoss(torch.nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pred, load):
        """Penalizes failure to allocate sufficient material in regions where load is applied.
        # taken from Tabarraei et al. "Physics-informed transformer for real-time high-fidelity topology optimization"
        
        load: tensor as provided by the dataset (x_location_normalized, y_location_normalized, x_magnitude, y_magnitude)
        """
        
        pred2d = pred.reshape(
            pred.shape[0], 
            64 // self.patch_size, 64 // self.patch_size, 
            self.patch_size, self.patch_size
        ).permute(0, 1, 3, 2, 4).reshape(pred.shape[0], 64, 64)
        
        # get elements adjacent to load application points
        adjacent_elements_rows, adjacent_elements_cols, valid_elements = _load_adjacent_elements(load)
        adjacent_elements = torch.stack((adjacent_elements_rows, adjacent_elements_cols), dim=-1)
        
        
        # half of the force magnitude for each load component
        batch_indices = torch.arange(pred.shape[0], device=pred.device)[:, None]
        pred_adjacent = pred2d[
            batch_indices,
            adjacent_elements[:, :, 0],
            adjacent_elements[:, :, 1],
        ].masked_fill(~valid_elements, 0)
        
        
        n_adjacent = valid_elements.sum(dim=1, keepdim=True).clamp_min(1)
        element_forces = load[:, None, 2:4] / n_adjacent.unsqueeze(-1)
        element_force_magnitudes = torch.linalg.vector_norm(pred_adjacent.unsqueeze(-1) * element_forces, ord=2, dim=-1)
        loss = 1 - element_force_magnitudes.sum(dim=1)
        return loss.mean() # average over batch
    


class FloatingMaterialLoss(torch.nn.Module):
    def __init__(self, patch_size, max_iteration=300, tolerance=1e-4, scale=10, shift=0.5):
        """
        Initializes the FloatingMaterialLoss module.

        Args:
            max_iteration: Maximum number of iterations for the floating material loss calculation.
            tolerance: Convergence tolerance for the floating material loss calculation. (1e-4 suggested in Tabarraei et al.)
            scale: Scale factor for the sigmoid function. (try: 5, 10, 20, ...)
            shift: Shift factor for the sigmoid function. (try: 0.25, 0.5, 0.75, ...)
        """
        
        super().__init__()
        self.patch_size = patch_size
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.scale = scale
        self.shift = shift

    def forward(self, pred, load):
        """Penalizes floating material not connected to the load.
        # taken from Tabarraei et al. "Physics-informed transformer for real-time high-fidelity topology optimization" -> Algorithm 1
        """
        
        pred_2d = pred.reshape(
            pred.shape[0], 
            64 // self.patch_size, 64 // self.patch_size, 
            self.patch_size, self.patch_size
        ).permute(0, 1, 3, 2, 4).reshape(pred.shape[0], 64, 64)
                
        # get seed as element adjacent to the load (which must always exist)
        rows, cols, valid = _load_adjacent_elements(load)
        batch = torch.arange(pred.shape[0], device=pred.device)
        # among the valid elements at the load, use the one with the highest predicted density
        densities = pred_2d[batch[:, None], rows, cols].masked_fill(~valid, -1)
        k = densities.argmax(dim=1)
        i_, j_ = rows[batch, k], cols[batch, k]
        
        H: torch.Tensor = torch.zeros_like(pred_2d, device=pred_2d.device)  # create a tensor of zeros with the same shape as pred_2d[0]
        H[torch.arange(pred.shape[0], device=pred.device), i_, j_] = 1.0
        S: torch.Tensor = torch.as_tensor([ # kernel for convolution
            [0, 0.5, 0],
            [0.5, 0.5, 0.5],
            [0, 0.5, 0]
        ], device=pred_2d.device)
        
        for i in range(self.max_iteration):
            H_prev = H
            H_neigh = torch.conv2d(H.unsqueeze(1), S.unsqueeze(0).unsqueeze(0), padding=1).squeeze(1) # conv2D expects shape (batch, channels, height, width)
            A = torch.sigmoid(self.scale*(H_neigh - self.shift))
            H = torch.maximum(H,  A*pred_2d) # todo review this, check mapping with pred 
            if i%10==0 and torch.max(H-H_prev) < self.tolerance:
                break
        
        # ! ai generated code snippet to help me out with the dimensions and transformations
        rows = torch.arange(64, device=pred_2d.device).view(1, 64, 1)
        cols = torch.arange(64, device=pred_2d.device).view(1, 1, 64)
        seed_mask = (rows == i_[:, None, None]) & (cols == j_[:, None, None])
        loss = ((pred_2d - H) * pred_2d).masked_fill(seed_mask, 0)
        # ! #########
        
        return loss.sum(dim=(1, 2)).mean()
    
class CustomLoss(torch.nn.Module):
    def __init__(self, base_loss_fn, patch_size, auxiliary_loss_weight=0.075):
        """
        Args:
            base_loss_fn: The base loss function to use.
            auxiliary_loss_weight: Weight for the auxiliary losses. Tabarraei et al. suggest 0.075
        """
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss_fn 
        self.auxiliary_weight = auxiliary_loss_weight
        self.n_elements = 64 * 64
        self.volume_fraction_loss = VolumeFractionLoss()
        self.load_discrepancy_loss = LoadDiscrepancyLoss(patch_size)
        self.floating_material_loss = FloatingMaterialLoss(patch_size)
        
    def forward(self, pred, target, load, target_volume_fraction):
        
        
        vf_loss = self.volume_fraction_loss(pred, target_volume_fraction)
        ld_loss = self.load_discrepancy_loss(pred, load) / self.n_elements # todo: review if this division by the number of elements is appropriate
        fm_loss = self.floating_material_loss(pred, load) / self.n_elements # example weights for vf_loss, fm_loss, ld_loss
        
        return self.base_loss(pred, target) + self.auxiliary_weight * (vf_loss + fm_loss + ld_loss) # all auxiliary losses weighted equally
    


# Copyright 2018 Daniel Dazac 
# MIT Licensed
# License and source: https://github.com/dfdazac/wassdistance/
# Slightly modified by Thomas Viehmann, see https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Pytorch_Wasserstein.ipynb
# Also slightly modifed by me to adapt to use case
# ! ai was used to improve and evaluate this implementation
class SinkhornDistance(torch.nn.Module):
    """
    Computes the entropy-regularized transport cost between distributions. 
    I.e. imagine moving something from one distribution to match another in the most efficient way possible.
    Depending on how much mass has to be moved and the distance it has to travel the transport cost will vary.
    The cost of moving one unit of mass from one location to another is defined by the (squared) distance between the locations.
    Integrating over all masses and journeys gives the total transport cost.
    
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        patch_size (int): patch size used to flatten the density maps
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
        tolerance (float): maximum allowed L1 marginal error for convergence
        check_every (int): check marginal error after this many iterations
    Shape:
        - Inputs: :math:`(N, 4096)` for both density maps
        - Output: entropic transport cost, :math:`(N)` for reduction='none', else scalar
    """
    def __init__(
        self, patch_size, eps=1e-2, max_iter=100, reduction='mean',
        tolerance=1e-3, check_every=10,
    ):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not isinstance(patch_size, int) or patch_size <= 0 or 64 % patch_size != 0:
            raise ValueError("patch_size must be positive and divide 64")
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not isinstance(check_every, int) or check_every <= 0:
            raise ValueError("check_every must be positive")

        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.tolerance = tolerance
        self.check_every = check_every
        self.n_elements = 64 * 64

        # map the patch-ordered flat index to the 64x64 grid position
        k = torch.arange(self.n_elements)
        n = 64 // patch_size
        p, q = k // patch_size**2, k % patch_size**2
        row = (p // n) * patch_size + q // patch_size
        col = (p % n) * patch_size + q % patch_size
        self.register_buffer("grid_to_patch", torch.argsort(row * 64 + col), persistent=False)

        # The squared euclidean cost on the grid is separable:
        # C[(i,j),(k,l)] = c[i,k] + c[j,l], so the dense (4096, 4096) matrix is never built.
        coords_1d = torch.arange(64).float() / 64 # normalized coordinates
        cost_1d = (coords_1d[:, None] - coords_1d[None, :]) ** 2
        self.register_buffer("C_1d", cost_1d, persistent=False)

    def forward(self, mu, nu):
        """
        Args:
            mu, nu: marginal distributions with shape (batch, 4096), 
            i.e. pred and target densities.
        """
        
        if mu.ndim != 2 or nu.ndim != 2:
            raise ValueError("mu and nu must have shape (batch, 4096)")
        if mu.shape != nu.shape or mu.shape[1] != self.n_elements:
            raise ValueError("mu and nu must have the same shape (batch, 4096)")
        if mu.device != nu.device or mu.dtype != nu.dtype:
            raise ValueError("mu and nu must have the same device and dtype")
        if mu.device != self.C_1d.device:
            raise ValueError("move SinkhornDistance to the same device as mu and nu")
        if not mu.is_floating_point() or not nu.is_floating_point():
            raise TypeError("mu and nu must be floating-point tensors")
        if not torch.isfinite(mu).all() or not torch.isfinite(nu).all():
            raise ValueError("mu and nu must contain only finite values")
        if (mu < 0).any() or (nu < 0).any():
            raise ValueError("mu and nu must be nonnegative")
        if (mu.sum(dim=-1) <= 0).any() or (nu.sum(dim=-1) <= 0).any():
            raise ValueError("each input map must contain positive total mass")

        # Normalize each map to unit mass; clamp gives zero-density pixels a tiny
        # positive mass so the logarithmic updates remain finite.
        mu = mu.clamp_min(1e-8)
        nu = nu.clamp_min(1e-8)
        mu = mu / mu.sum(dim=-1, keepdim=True)
        nu = nu / nu.sum(dim=-1, keepdim=True)

        batch_size = mu.shape[0]

        a = mu[:, self.grid_to_patch].view(batch_size, 64, 64)
        b = nu[:, self.grid_to_patch].view(batch_size, 64, 64)

        # Envelope theorem: at the optimum the gradient of the entropic OT cost w.r.t. a is
        # the dual potential u, so the iterations need no autograd graph (saves memory).
        with torch.no_grad():
            u, v = self._sinkhorn(a.detach(), b.detach())

        # dual objective <u, a> + <v, b>: equals the entropic OT cost, gradient flows through a and b
        cost = (u * a).sum(dim=(-2, -1)) + (v * b).sum(dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def _sinkhorn_step(self, a, b, u, v):
        # u_ij = eps * (log a_ij - LSE_kl[(-C[(i,j),(k,l)] + v_kl) / eps])
        u = self.eps * (torch.log(a) - self._log_kernel(v))
        v = self.eps * (torch.log(b) - self._log_kernel(u))
        return u, v

    def _sinkhorn(self, mu, nu):
        """Returns the dual potentials u, v (log-domain Sinkhorn iterations)."""
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        return self._iterate(mu, nu, u, v)

    def _iterate(self, mu, nu, u, v):
        converged = False
        marginal_error = float('inf')
        actual_nits = 0

        for i in range(self.max_iter):
            u, v = self._sinkhorn_step(mu, nu, u, v)

            actual_nits = i + 1
            if actual_nits % self.check_every == 0 or actual_nits == self.max_iter:
                with torch.no_grad():
                    # marginals of the transport plan pi = exp((-C + u + v) / eps)
                    row_marginal = torch.exp(u / self.eps + self._log_kernel(v))
                    col_marginal = torch.exp(v / self.eps + self._log_kernel(u))
                    row_error = (row_marginal - mu).abs().sum(dim=(-2, -1))
                    col_error = (col_marginal - nu).abs().sum(dim=(-2, -1))
                    marginal_error = torch.maximum(row_error, col_error).max().item()
                if marginal_error <= self.tolerance:
                    converged = True
                    break

        self.actual_nits = actual_nits
        self.converged = converged
        self.marginal_error = marginal_error
        if not converged:
            warnings.warn(
                f"Sinkhorn did not reach marginal tolerance {self.tolerance:g} "
                f"after {self.max_iter} iterations (error={marginal_error:.3g})",
                RuntimeWarning,
            )
        return u, v

    def _log_kernel(self, v):
        """
        Separable log-domain kernel application for potentials of shape (batch, 64, 64):
        out_ij = LSE_kl[(-c_ik - c_jl + v_kl) / eps] = LSE_l[-c_jl / eps + LSE_k[(-c_ik + v_kl) / eps]]
        Peak memory is (batch, 64, 64, 64) instead of (batch, 4096, 4096).
        """
        log_k = -self.C_1d / self.eps
        t = torch.logsumexp(log_k[None, :, :, None] + v[:, None, :, :] / self.eps, dim=2) # (b, i, l)
        return torch.logsumexp(log_k[None, None, :, :] + t[:, :, None, :], dim=-1) # (b, i, j)

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1