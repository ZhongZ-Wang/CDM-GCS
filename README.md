# Geological CO2 storage
Geological CO2 storage (GCS) is expected to play a pivotal role in achieving climate-neutrality targets by 2050. Accurate prediction of long-term CO2 storage performance relies on inverse modeling proce-
dures that precisely characterize spatially varying geological properties using practically available observed data.

Schematic diagram：

![](https://github.com/ZhongZ-Wang/CDM-GCS/blob/main/Fig/gcs.png)


## CDM4GCS: Generative inverse modeling for GCS prediction via conditional diffusion models

We present an end-to-end generative inversion framework based on the conditional diffusion model for efficiently characterizing heterogeneous geological properties and accelerating the inversion process. By employing an improved U-net to learn the conditional denoising diffusion process, the proposed framework enables the direct generation of high-dimensional property fields that closely match the observed data, eliminating the need for iterative forward simulations. Additionally, the probabilistic nature inherent in the diffusion approach allows for producing an ensemble of plausible geological realizations, facilitating effective quantification of parametric and predictive uncertainties.

Framework:

![](https://github.com/ZhongZ-Wang/CDM-GCS/blob/main/Fig/cdm.png)

Network architecture:

![](https://github.com/ZhongZ-Wang/CDM-GCS/blob/main/Fig/network.png)

### Datasets
The datasets used in CDM have been uploaded to Google Drive and can be downloaded using the following links:

2D dataset: [Google Drive](https://drive.google.com/drive/folders/1eHh9clEqTyWPladjSMiKaqqRJFWx-ze8).

3D dataset: [Google Drive](https://drive.google.com/drive/folders/1I7bBZDVjF2Xet5dwdI7Cg1s0MvPkdgMP).

### Pretrained model

2D: [Google Drive](https://drive.google.com/drive/folders/1G57_DJLf5lh5Eq2LVci0Tg5KkmxYgLKu).

3D: [Google Drive](https://drive.google.com/drive/folders/1RpqtgVACs42692tVjEQV_lKb4WcwHhyk).

### Network Training
```
python Train.py
```

### Network Inference (prediction)
```
python Inference.py
```

## Supplementary Material

Supplementary information: ![](https://github.com/ZhongZ-Wang/CDM-GCS/tree/main/SM/Supplementary Material.pdf).
