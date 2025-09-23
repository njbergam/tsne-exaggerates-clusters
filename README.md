Analyzing the robustness of the t-SNE data visualization method. 

0_basics.ipynb
+ running and analyzing the training dynamics of the t-SNE run on basic input affinity matrices

1_false_positives.ipynb
+ how t-sne and umap take highly ``unclustered'' data and depict it as clustered

2_perturbation_stability.ipynb
+ demonstrating the sensitivity of t-SNE outputs on high-aspect ratio
+ exploring perturbation in both an average sense and an adversarial sense

3_outlier_behavior.ipynb
+ witnessing how t-sne and umap misrepresent outliers

functions.py
+ contains many of the functions used throughout the demonstrations


To start the virtual environment (``umap_env'') use the following commands
    source umap_env/bin/activate