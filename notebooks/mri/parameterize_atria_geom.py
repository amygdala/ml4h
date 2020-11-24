# %%
import vtk
import h5py
import time
import glob
import vtk
from vtk.util import numpy_support as ns
import sys
import pandas as pd
import numpy as np
from notebooks.mri.mri_atria import to_xdmf
from parameterize_segmentation import annotation_to_poisson
from ml4cvd.tensor_from_file import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4cvd.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES

def normal_to_dataset(ds):
    centers = vtk.vtkCellCenters()
    centers.SetInputData(ds)
    centers.Update()

    center_coors = ns.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
    sample = [0, -1, 200]
    v1 = center_coors[sample[1]] - center_coors[sample[0]]
    v2 = center_coors[sample[2]] - center_coors[sample[0]]
    n = np.cross(v1, v2) 
    n = n / np.linalg.norm(n)
    return n

def angle_between_datasets(ds1, ds2):
    n1 = normal_to_dataset(ds1)
    n2 = normal_to_dataset(ds2)
    angle = np.arccos(np.dot(n1, n2))
    return np.degrees(angle)


# %%
hd5s = glob.glob('/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/*.hd5')

# %%
start = int(sys.argv[1])
end = int(sys.argv[2])

# %%
views = ['3ch', '2ch', '4ch']
view_format_string = 'cine_segmented_lax_{view}'
annot_format_string = 'cine_segmented_lax_{view}_annotated'
annot_time_format_string = 'cine_segmented_lax_{view}_annotated_{t}'

channels = [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
            MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LA_cavity'], 
            MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']]

petersen = pd.read_csv('/home/pdiachil/returned_lv_mass.tsv', sep='\t')
petersen = petersen.dropna()

# %%
start_time = time.time()
results = {f'LA_poisson_{t}': [] for t in range(MRI_FRAMES)}
results['sample_id'] = []
for i, hd5 in enumerate(sorted(hd5s)):
    hd5 = f'/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/{start}.hd5'    
    sample_id = hd5.split('/')[-1].replace('.hd5', '')
    # if i < start:
    #     continue
    # if i == end:
    #     break
    
    annot_datasets = []
    orig_datasets = []
    try:
        with h5py.File(hd5) as ff_trad:
            for view in views:
                annot_datasets.append(_mri_hd5_to_structured_grids(ff_trad, annot_format_string.format(view=view),
                                                                view_name=view_format_string.format(view=view), 
                                                                concatenate=True, annotation=True,
                                                                save_path=None, order='F')[0])
                orig_datasets.append(_mri_hd5_to_structured_grids(ff_trad, view_format_string.format(view=view),
                                                                    view_name=view_format_string.format(view=view), 
                                                                    concatenate=False, annotation=False,
                                                                    save_path=None, order='F')[0])
                to_xdmf(annot_datasets[-1], f'{start}_{view}_annotated')
                to_xdmf(orig_datasets[-1], f'{start}_{view}_original')
            print(f'angle 3ch-2ch: {angle_between_datasets(orig_datasets[0], orig_datasets[1]):.1f}')
            print(f'angle 3ch-4ch: {angle_between_datasets(orig_datasets[0], orig_datasets[2]):.1f}')
            print(f'angle 4ch-2ch: {angle_between_datasets(orig_datasets[2], orig_datasets[1]):.1f}')
            poisson_atria, poisson_volumes = annotation_to_poisson(annot_datasets, channels, views, annot_time_format_string, range(MRI_FRAMES))

            results['sample_id'].append(sample_id)
            for t, poisson_volume in enumerate(poisson_volumes):
                results[f'LA_poisson_{t}'].append(poisson_volume/1000.0)

            for t, atrium in enumerate(poisson_atria):
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetInputData(atrium)                                        
                writer.SetFileName(f'/home/pdiachil/projects/atria/poisson_atria_{sample_id}_{t}.vtp')
                writer.Update()
    except:
        print('exception')
        break
    break

results_df = pd.DataFrame(results)
results_df.to_csv(f'atria_processed_{start}_{end}.csv')
end_time = time.time()
print(end_time-start_time)