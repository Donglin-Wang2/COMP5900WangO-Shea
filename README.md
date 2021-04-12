# Instructions on running the code
1. Make sure you have the LFSD folder set up:
   1. Go to this website: https://www.eecis.udel.edu/~nianyi/LFSD.htm and download all data packages. These should be raw_images.rar, focus_stack.rar, all_focus_images.rar, ground_truth.rar, and depth_map.rar.
   2. Extract all of these compressed files. You will get folders with the same name as the compressed files.
   3. But them all under one folder titled "LFSD" so that your "LFSD" folder structure looks like this:
   
        ```
        LFSD
        ├── all_focus_images
        ├── depth_map
        ├── focus_stack
        ├── ground_truth
        └── raw_images
        ```

2. Make sure that you create three empty folders titled ./results, ./data, and ./images. 
3. Maks sure that ./vae_vanilla, ./vae_adapted, ./vae_injected are under both the ./images and ./results folder. Afterward, you should have the ./images and ./results folder should look like this:

        ```
        images
        ├── vae_adapted
        ├── vae_injected
        └── vae_vanilla
        results
        ├── vae_adapted
        ├── vae_injected
        └── vae_vanilla
        ```
        
4. Run data_processing.py
5. Run LFSD_feat_gen.py
6. Run CIFAR100_feat_gen.py
7. Run vae_vanilla.ipynb, vae_adapted.ipynb, vae_injected.ipynb