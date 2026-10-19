# todo: implement custom loss function

# auxiliary losses ideas:
# - non-binary densities (encourage densities to be close to 0 or 1)
# - floating material


import torch


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
    def __init__(self):
        super().__init__()

    def forward(self, pred, load):
        """Penalizes failure to allocate sufficient material in regions where load is applied.
        # taken from Tabarraei et al. "Physics-informed transformer for real-time high-fidelity topology optimization"
        
        load: tensor as provided by the dataset (x_location_normalized, y_location_normalized, x_magnitude, y_magnitude)
        """
        
        pred2d = pred.reshape(-1, 8, 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(-1, 64, 64)
        
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
    def __init__(self, max_iteration=300, tolerance=1e-4, scale=10, shift=0.5):
        """
        Initializes the FloatingMaterialLoss module.

        Args:
            max_iteration: Maximum number of iterations for the floating material loss calculation.
            tolerance: Convergence tolerance for the floating material loss calculation. (1e-4 suggested in Tabarraei et al.)
            scale: Scale factor for the sigmoid function. (try: 5, 10, 20, ...)
            shift: Shift factor for the sigmoid function. (try: 0.25, 0.5, 0.75, ...)
        """
        
        super().__init__()
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.scale = scale
        self.shift = shift

    def forward(self, pred, load):
        """Penalizes floating material not connected to the load.
        # taken from Tabarraei et al. "Physics-informed transformer for real-time high-fidelity topology optimization" -> Algorithm 1
        """
        
        pred_2d = pred.reshape(-1, 8, 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(-1, 64, 64) # todo: review this
        
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
    def __init__(self, base_loss_fn, auxiliary_loss_weight=0.075):
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
        self.load_discrepancy_loss = LoadDiscrepancyLoss()
        self.floating_material_loss = FloatingMaterialLoss()
        
    def forward(self, pred, target, load, target_volume_fraction):
        
        
        vf_loss = self.volume_fraction_loss(pred, target_volume_fraction)
        ld_loss = self.load_discrepancy_loss(pred, load) / self.n_elements # todo: review if this division by the number of elements is appropriate
        fm_loss = self.floating_material_loss(pred, load) / self.n_elements # example weights for vf_loss, fm_loss, ld_loss
        
        return self.base_loss(pred, target) + self.auxiliary_weight * (vf_loss + fm_loss + ld_loss) # all auxiliary losses weighted equally
    
