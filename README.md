# Geological CO2 storage
Geological CO2 storage (GCS) is expected to play a pivotal role in achieving climate-neutrality targets by 2050. Accurate prediction of long-term CO2 storage performance relies on inverse modeling proce-
dures that precisely characterize spatially varying geological properties using practically available observed data.

![Schematic diagram](https://github.com/ZhongZ-Wang/CDM-GCS/blob/main/Fig/gcs.png)


# CDM-GCS: Generative inverse modeling for improved geological CO2 storage prediction via conditional diffusion models

We present an end-to-end generative inversion framework based on the conditional diffusion model for efficiently characterizing heterogeneous geological properties and accelerating the inversion process. By employing an improved U-net to learn the conditional denoising diffusion process, the proposed framework enables the direct generation of high-dimensional property fields that closely match the observed data, eliminating the need for iterative forward simulations. Additionally, the probabilistic nature inherent in the diffusion approach allows for producing an ensemble of plausible geological realizations, facilitating effective quantification of parametric and predictive uncertainties.

Framework:

![Framework](https://github.com/ZhongZ-Wang/CDM-GCS/blob/main/Fig/cdm.png)

Network architecture:

![Network architecture](https://github.com/ZhongZ-Wang/CDM-GCS/blob/main/Fig/unet.png)

# Datasets
The datasets used in CDM have been uploaded to Google Drive and can be downloaded using the following links:

2D dataset: [Google Drive](https://drive.google.com/drive/folders/1eHh9clEqTyWPladjSMiKaqqRJFWx-ze8).

3D dataset: [Google Drive](https://drive.google.com/drive/folders/1I7bBZDVjF2Xet5dwdI7Cg1s0MvPkdgMP).

# Network Training
```
python Train.py
```

# Network Inference (prediction)
```
python Inference.py
```

## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@article{Chen2024,
  title={Machine Learning-Accelerated Discovery of Optimal Multi-Objective Fractured Geothermal System Design},
  author={Guodong Chen and Jiu Jimmy Jiao and Qiqi Liu and Zhongzheng Wang and Yaochu Jin},
  year={2024}
}
```
