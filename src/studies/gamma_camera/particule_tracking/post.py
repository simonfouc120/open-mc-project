import h5py
import numpy as np

def load_data_tracks():
    with h5py.File('tracks.h5', 'r') as f:
        track_names = sorted(f.keys())
        for name in track_names:
            print(f"Track: {name}")
            data = f[name][:]
            print("  r (positions):", data['r'])
            print("  E (energies):", data['E'])
            print()



tracks_data = {}

with h5py.File('tracks.h5', 'r') as f:
    for idx, name in enumerate(sorted(f.keys())):
        data = f[name][:]
        tracks_data[name] = {
            'r': np.array(data['r']),
            'u': np.array(data['u']),
            'E': np.array(data['E']),
            'time': np.array(data['time']),
            'wgt': np.array(data['wgt']),
            'cell_id': np.array(data['cell_id']),
            'cell_instance': np.array(data['cell_instance']),
            'material_id': np.array(data['material_id']),
            'particle_number': idx+1  # Add particle number as index
        }

# Example: access particle number of track_1_1_1
print(tracks_data['track_1_1_1']['particle_number'])
# Example: access all positions of track_1_1_1
print(tracks_data['track_1_1_1']['r'])
print(tracks_data['track_1_1_1']['E'])
print(tracks_data['track_1_1_1']['time'])
print(tracks_data['track_1_1_1']['cell_id'])
print(tracks_data['track_1_1_1']['material_id'])