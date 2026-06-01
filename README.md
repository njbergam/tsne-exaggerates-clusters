# t-SNE exaggerates clusters
Exploring failure modes of t-SNE. Code for paper: https://arxiv.org/abs/2510.07746

## Interactive demo (GitHub Pages)

Two demo modes (**outlier** vs **poison point**): set parameters, view a **PCA** preview (clusters colored, injection in red), then press **Start t-SNE** to run gradient descent until convergence.

- **Live site:** enable [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-pages-site-for-your-repository) for this repo with source **`/docs`** (folder on `main`). The URL will be `https://<your-username>.github.io/tsne-exaggerates-clusters/`.
- **Local preview:** `python -m http.server 8080 --directory docs` then open http://localhost:8080

Pushes to `main`/`master` that touch `docs/` deploy automatically via `.github/workflows/pages.yml`.

## Python notebooks

To run the code, please create a python virtual environment and install the requirements.

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


0: Demo (0_demo.ipynb)

1: Impostor of PCMC3k Single-Cell Data (1_impostor_singleCell.ipynb)

2: Adversarial Perturbations of Low-Aspect Ratio Data (2_adversarial.ipynb)

3: Alpha-Outliers on Synthetic Data (3_outlier_synthetic.ipynb)

4: Alpha-Outliers re: Credit Fraud Identification (4_outlier_creditFraud.ipynb)

5: Poison Points vs Outlier Points on BBC News Data (5_outlier+poison_bbc.ipynb)