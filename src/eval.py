"""Create evaluations for the results of the vision transformer model"""


# define evaluation functions for the vision transformer model
def compliance_error(predicted, target):
    
    # todo: how do we actually compute the compliance????
    
    pass

def density_error(predicted, target):
    # todo: how do we define density error?
    
    pass

def floating_material_error(predicted, target):
    # todo: how do we define floating material error?
    pass

def volume_error(predicted, target):
    """Compute the absolute error in material volume fraction between the predicted and target topologies"""
    
    target_volume = target.sum() / target.numel()
    predicted_volume = predicted.sum() / predicted.numel()
    
    return abs(predicted_volume - target_volume) # between 0 and 1

def evaluate(predicted, target):
    
    # the purpose of this model is to accelerat topology optimization
    # to evaluate the results we have to evaluate
    # - physicality: densities other then 0 or 1 should be minimized
    #   -> we measure density and floating material
    # - function: (as in inverse of stiffness). Since the goal of this type of topology optimization is to 
    #   minimize compliance for a given material volume, the predicted topology should have a compliance close to the target compliance.
    #   -> we measure compliance error
    # - volume: the predicted topology should respect the target material volume.
    #   -> we measure volume error
    
    return {
        "compliance_error": compliance_error(predicted, target),
        "density_error": density_error(predicted, target),
        "floating_material_error": floating_material_error(predicted, target),
        "volume_error": volume_error(predicted, target)
    }