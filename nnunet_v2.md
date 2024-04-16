## Configuration Environment
```
conda create --name nnunet_v2 python=3.9 -y
conda activate nnunet_v2

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2

```

```
set nnUNet_raw=Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_raw
set nnUNet_preprocessed=Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_preprocessed
set nnUNet_results=Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_results
```


## Convert MSD dataset
```
nnUNetv2_convert_MSD_dataset -i Z:\Nana\TransUnet_For_FINDS\data\DATASET\Task500_FINDS
nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity
```

## train
```
nnUNetv2_train 500 3d_lowres 0 --npz
nnUNetv2_train 500 3d_fullres 1 --npz
nnUNetv2_train 500 3d_cascade_fullres 2 --npz
```
### Set GPU
```
set CUDA_VISIBLE_DEVICES=1
```

### Set CPU
```
set OMP_NUM_THREADS=1 
```


### val npz
```
nnUNetv2_train 500 3d_fullres 0 --val --npz
nnUNetv2_train 500 3d_fullres 1 --val --npz
nnUNetv2_train 500 3d_fullres 2 --val --npz
nnUNetv2_train 500 3d_fullres 3 --val --npz
nnUNetv2_train 500 3d_fullres 4 --val --npz
```


### find best configuration
```
nnUNetv2_find_best_configuration 500 -c 3d_fullres 3d_lowres 3d_cascade_fullres  

```

### predict
```
# Nature
nnUNetv2_predict -i Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_raw\Dataset500_FINDS\imagesTs -o Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_cascade_fullres -d 500 -c 3d_cascade_fullres -num_parts 4 -part_id 0 --save_probabilities -prev_stage_predictions Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_lowres

set CUDA_VISIBLE_DEVICES=1 
nnUNetv2_predict -i Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_raw\Dataset500_FINDS\imagesTs -o Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_cascade_fullres -d 500 -c 3d_cascade_fullres -num_parts 4 -part_id 1 --save_probabilities -prev_stage_predictions Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_lowres

# IPAC
nnUNetv2_predict -i Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_raw\Dataset500_FINDS\imagesTs -o Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_cascade_fullres -d 500 -c 3d_cascade_fullres -num_parts 4 -part_id 2 --save_probabilities -prev_stage_predictions Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_lowres

# Steam
nnUNetv2_predict -i Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_raw\Dataset500_FINDS\imagesTs -o Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_cascade_fullres -d 500 -c 3d_cascade_fullres -num_parts 4 -part_id 3 --save_probabilities -prev_stage_predictions Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_lowres

```


### ensemble
```
nnUNetv2_ensemble -i Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_lowres Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_cascade_fullres -o Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\ensemble 
```

## postprocessing

```
nnUNetv2_apply_postprocessing -i Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_cascade_fullres -o Z:\Nana\TransUnet_For_FINDS\data\Prediction\Dataset500_FINDS\3d_cascade_fullres_pp -pp_pkl_file Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_results\Dataset500_FINDS\nnUNetTrainer__nnUNetPlans__3d_cascade_fullres\crossval_results_folds_0_1_2_3_4\postprocessing.pkl -np 8 -plans_json Z:\Nana\TransUnet_For_FINDS\data\DATASET\nnUNet\nnUNet_results\Dataset500_FINDS\nnUNetTrainer__nnUNetPlans__3d_cascade_fullres\crossval_results_folds_0_1_2_3_4\plans.json
```


