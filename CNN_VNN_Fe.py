from pymatgen.io.vasp import Poscar
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
import numpy as np
import pandas as pd

# Load the structure
structure = Poscar.from_file("POSCAR_Fe").structure

# Add oxidation states
oxidation_states = {'Fe': +2, 'Ge': -2}
structure.add_oxidation_state_by_element(oxidation_states)

# Initialize CrystalNN and VoronoiNN
cnn = CrystalNN(cation_anion=False, distance_cutoffs=(0, 3.0))  # set cut-off to 3 Angstrom
vnn = VoronoiNN(allow_pathological=True)

# Prepare a list to collect the data for each site
data = []

# Calculate nearest neighbors and Voronoi polyhedra
for n, site in enumerate(structure):
    site_distances = []
    site_volumes = []

    # Nearest neighbors
    neighbors = cnn.get_nn_info(structure, n)
    for neighbor in neighbors:
        # Calculate the actual distance between the site and its neighbor
        neighbor_site = structure[neighbor['site_index']]
        distance = site.distance(neighbor_site)
        site_distances.append(distance)
    
    # Voronoi polyhedra
    voronoi_polyhedra = vnn.get_voronoi_polyhedra(structure, n)
    for _, polyhedron in voronoi_polyhedra.items():
        volume = polyhedron.get('volume')
        if volume is not None:
            site_volumes.append(volume)
    
    # Calculate statistics for this site
    min_distance = np.min(site_distances) if site_distances else None
    max_distance = np.max(site_distances) if site_distances else None
    mean_distance = np.mean(site_distances) if site_distances else None

    mean_volume = np.mean(site_volumes) if site_volumes else None
    std_dev_volume = np.std(site_volumes) if site_volumes else None

    coordination_number = len(neighbors)

    # Collect data for this site
    data.append([min_distance, max_distance, mean_distance, mean_volume, std_dev_volume, coordination_number])

# Create a DataFrame from the collected data
df = pd.DataFrame(data, columns=['Min distance (Å)', 'Max distance (Å)', 'Mean distance (Å)', 'Mean volume (Å³)', 'Volume std. dev. (Å³)', 'Coordination number'])
print(df)
df.to_csv("CNN_VNN_Fe.txt", sep="\t", index=False)