def calculate_reynolds_number(density, velocity, characteristic_length, viscosity):
    """
    Calculate the Reynolds number for the fluid flow.

    Parameters:
        density (float): Fluid density (kg/m^3).
        velocity (float): Fluid velocity (m/s).
        characteristic_length (float): Characteristic length of the body (m).
        viscosity (float): Dynamic viscosity of the fluid (Pa·s).

    Returns:
        float: Reynolds number.
    """
    return (density * velocity * characteristic_length) / viscosity

def calculate_drag_coefficient(reynolds_number):
    """
    Estimate drag coefficient from Reynolds number for a streamlined body.

    Parameters:
        reynolds_number (float): Reynolds number.

    Returns:
        float: Estimated drag coefficient.
    """
    re = max(float(reynolds_number), 1e-8)

    if re < 1.0:
        return 24.0 / re
    if re < 1000.0:
        return 24.0 / re * (1.0 + 0.15 * (re ** 0.687))
    if re < 2.0e5:
        return 0.44
    return 0.2

def calculate_gradient_coefficients(shape, material):
    """
    Calculate gradient coefficients based on the shape and material properties.

    Parameters:
        shape (str): Shape of the body (e.g., "cylinder", "sphere").
        material (str): Material of the body (e.g., "aluminum", "steel").

    Returns:
        dict: Gradient coefficients for drag, lift, etc.
    """
    # Example coefficients (these would be determined experimentally or from literature)
    coefficients = {
        "cylinder": {"drag": 1.2, "lift": 0.3},
        "sphere": {"drag": 0.5, "lift": 0.1}
    }
    material_factors = {
        "aluminum": 1.0,
        "steel": 1.2
    }

    if shape in coefficients and material in material_factors:
        base_coeffs = coefficients[shape]
        factor = material_factors[material]
        return {key: value * factor for key, value in base_coeffs.items()}
    else:
        raise ValueError("Invalid shape or material")