# Cancer omics machine learning

![Heatmap of the PAM50 genes selected by the model, across training samples grouped by breast cancer subtype](docs/figures/pam50_heatmap.png)

![Final model: nested cross-validation macro F1 against a majority baseline, with the test-set confusion matrix](docs/figures/final_model.png)

![Macro F1 for each omics type and combination across five cross-validation folds](docs/figures/multi_omics_scores.png)

![PCA of all mRNA genes for the training samples, coloured by PAM50 subtype](docs/figures/pca_subtypes.png)

## Code

```mermaid
flowchart TD
    data[("MLOmics GS-BRCA<br/>mRNA · miRNA · methylation · CNV")]

    lc["<b>learn_classifier.py</b><br/>load and audit · stratified split<br/>majority baseline · tuned logistic regression"]
    sf["<b>select_features.py</b><br/>ANOVA gene selection inside each fold<br/>shuffled-label leakage test"]
    lt["<b>learn_torch.py</b><br/>linear and hidden-layer PyTorch models<br/>early stopping"]
    cm["<b>compare_models.py</b><br/>paired fold comparison<br/>scikit-learn vs PyTorch"]
    mo["<b>multi_omics.py</b><br/>single and combined omics types"]
    im["<b>inspect_model.py</b><br/>model weights · PAM50 overlap"]
    fm["<b>final_model.py</b><br/>nested cross-validation<br/>one test-set evaluation"]
    pca["<b>pca_plot.py</b><br/>unsupervised PCA"]
    pt["<b>pytorch_ml.py</b><br/>standalone PyTorch lesson"]

    f1[/"pam50_heatmap.png"/]
    f2[/"final_model.png"/]
    f3[/"multi_omics_scores.png"/]
    f4[/"pca_subtypes.png"/]

    data --> lc
    data --> mo
    data --> pt
    lc --> sf
    lc --> lt
    lt --> cm
    lc --> cm
    lc --> mo
    sf --> im
    sf --> fm
    im --> fm
    im --> pca
    lc --> pca

    im --> f1
    fm --> f2
    mo --> f3
    pca --> f4
```

## Data

The input data come from **MLOmics**, using the `GS-BRCA` classification
dataset: 671 TCGA breast tumours with mRNA, miRNA, DNA methylation and copy
number measurements, each labelled with one of five PAM50 subtypes (LumA, Her2,
LumB, Normal, Basal).

- Dataset: [MLOmics on Hugging Face](https://huggingface.co/datasets/AIBIC/MLOmics) (CC BY 4.0)
- Benchmark code: [Cancer Multi-Omics Benchmark](https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark)
- Paper: Yang, Z. *et al.* MLOmics: Cancer Multi-Omics Database for Machine
  Learning. *Scientific Data* **12** (2025).
  [doi:10.1038/s41597-025-05235-x](https://doi.org/10.1038/s41597-025-05235-x)

The data are not included in this repository; see
[data/README.md](data/README.md) for where to place them.
