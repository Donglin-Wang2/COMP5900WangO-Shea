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

2. Make sure that you create three empty folders titled ./results, ./data, ./plots, and ./images. 
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

4. Run data_processing.py. This process the content of the ./LFSD folder into the propser format.
5. Run LFSD_feat_gen.py. This generates the features produced by the ImageNet pre-trained models on the LFST data.
6. Run CIFAR100_feat_gen.py. This generates the features produced by the ImageNet pre-trained models on the CIFAR100 data.
7. Run vae_vanilla.ipynb, vae_adapted.ipynb, vae_injected.ipynb. This trains the 12 model architectures discussed in the report.
8. Run CIFAR100_efficientnet_comparison.ipynb, CIFAR100_inception_comparison.ipynb, CIFAR100_mobilenet_comparison.ipynb, and CIFAR100_resnet_comparison.ipynb. This trains the encoding network in the previous step to be used in classification.
9. Run plot_reuslts.ipynb