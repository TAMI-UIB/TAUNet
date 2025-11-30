# TAUNet: Transformer Assisted UNet for Marine Litter Detection on Sentinel-2 Imagery

This repository contains the implementation and additional resources for the following paper:

**Transformer Assisted UNet for Marine Litter Detection on Sentinel-2 Imagery**  
*Daniel Torres, Bartomeu Garau, Francesc Alcover, Ivan Pereira-Sánchez, Julia Navarro, Catalina Sbert, Joan Duran*  

[![EarthArXiv](https://img.shields.io/badge/EarthArXiv-10424-green.svg)](https://eartharxiv.org/repository/view/10424/)

---

## 📄 Abstract
The contamination of marine environments with man-made litter is a growing nation-wide concern. Satellite imagery combined with deep learning–based detection models has emerged as a robust and cost-effective solution for large-scale marine litter monitoring. In this article, we present a novel deep learning-based scheme to detect marine litter using Sentinel-2 imagery based on the Deep UNet architecture, introducing self- and cross-attention mechanisms into the decoder via transformer layers. The model leverages all Sentinel-2 bands except B10, and the NDVI and FDI indices are additionally incorporated to better guide the segmentation process. To evaluate the proposed method, we train it on the FloatingObjects dataset, a widely used benchmark for marine debris detection, and the results show that it compares favorably against state-of-the-art approaches.

<!--
---

## 📚 EarthArXiv Preprint

The paper is currently under revision, and the first preprint is available on [EarthArXiv](https://eartharxiv.org/repository/view/10424/).


---
-->

## 🛠️ Environment

You can set up the development environment using either **Conda** or **pip**.

#### 📦 Option 1: Using Conda (`environment.yml`)

1. Create the environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate TAUNet
   ```

---

#### 💡 Option 2: Using pip (`requirements.txt`)

1. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
---

## ⚙️ Setup

To begin, create an .env file in the project root directory and define the `DATASET_PATH` variable, pointing to the directory where your dataset is stored.

The DataModule is built specifically for FloatingObjects dataset.

---
## Train

Run the following command:
   ```bash
   python train.py 
   ```
---
## 🏗️ To-do's:
- [ ] Upload definitive checkpoints.
<!--
---
## 📌 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{torres2025transformer,
  title={Transformer Assisted U-Net for Marine Litter Detection on Sentinel-2 Imagery},
  author={Torres, Daniel and Garau, Bartomeu and Alcover, Francesc and Pereira-S{\'a}nchez, Ivan and Navarro, Julia and Sbert, Catalina and Duran, Joan},
  journal={EarthArXiv eprints},
  year={2025}
}
```
-->
