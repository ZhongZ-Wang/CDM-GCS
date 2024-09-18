# CDM-GCS: Generative inverse modeling for improved geological CO2 storage prediction via conditional diffusion models

We present an end-to-end generative inversion framework based on the conditional diffusion model for efficiently characterizing heterogeneous geological properties and accelerating the inversion process. By employing an improved U-net to learn the conditional denoising diffusion process, the proposed framework enables the direct generation of high-dimensional property fields that closely match the observed data, eliminating the need for iterative forward simulations. Additionally, the probabilistic nature inherent in the diffusion approach allows for producing an ensemble of plausible geological realizations, facilitating effective quantification of parametric and predictive uncertainties.

Framework:

![Framework](https://github.com/ZhongZ-Wang/CDM-GCS/tree/main/Fig/cdm.pdf "Framework")

Network architecture:

![Network architecture](https://github.com/ZhongZ-Wang/CDM-GCS/tree/main/Fig/unet.pdf "system design")

# Datasets
The datasets used in CDM have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1mi9Cmgnufi3kSMCeedP7G_K-4aEcd3_A?usp=drive_link) and can be downloaded using this link.

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
