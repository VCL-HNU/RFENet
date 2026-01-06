
""" Create a mesh template and store it """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"
import sys 
sys.path.append(".") 
from data.cortex import Cortex

structure_type = ('white_matter', 'cerebral_cortex')

# Important!!! (defines coordinate normalization in the template)
patch_size = [128, 144, 128]
select_patch_size = [192, 208, 192]
patch_origin = [0, 0, 0] # Not used here
n_vertices = 42016 #40962

template_path = f"/new_templates/cortex_4_TYPE_{n_vertices}_sps{select_patch_size}_ps{patch_size}.obj"

split = {
    'train': ['template'], # Use this ID for template creation
    'validation': [],
    'test': []
}

print("Creating dataset...")
dataset, _, _ = Cortex.split(raw_data_dir="/data/cotex/",
                             augment_train=False,
                             save_dir="/misc/",
                             fixed_split=split,
                             patch_origin=patch_origin,
                             select_patch_size=select_patch_size,
                             patch_size=patch_size,
                             structure_type=structure_type,
                             mesh_target_type='mesh',
                             reduced_freesurfer=0.5, #0.3,
                             n_ref_points_per_structure=10000, # irrelevant
                             mesh_type='freesurfer',
                             patch_mode="no")
print("Dataset created.")
print("Creating template...")

# Choose here the function that creates the desired template
path = dataset.store_specific_template(template_path, n_max_points=(1.0-n_vertices/51037.0)) #store_spacific_template store_index0_template

if path is not None:
    print("Template stored at " + path)

#CUDA_VISIBLE_DEVICES=4 python scripts/create_template_and_store.py