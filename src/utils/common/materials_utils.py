import openmc


def get_mass_fraction(material, nuclide):
    """
    Compute the mass fraction of a given nuclide in an OpenMC material.

    Parameters
    ----------
    material : openmc.Material
        The material to analyze.
    nuclide : str
        The nuclide name, e.g., "U235".

    Returns
    -------
    float
        The mass fraction of the nuclide in the material.
    """
    import openmc.data

    atom_densities = material.get_nuclide_atom_densities()
    total_mass = 0.0
    for nuc, atom_frac in atom_densities.items():
        total_mass += atom_frac * openmc.data.atomic_mass(nuc)
    if nuclide not in atom_densities:
        raise ValueError(f"{nuclide} not found in material.")
    mass_fraction = (atom_densities[nuclide] * openmc.data.atomic_mass(nuclide)) / total_mass
    return mass_fraction