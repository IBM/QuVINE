## QuVINE: Quantum-enabled View-Integrated Network Embeddings

[![QuVINE Framewor][quvine]](#)

We introduced QuVINE, a quantum-enhanced multi-view network embedding framework designed to address the inherent complexity and heterogeneity of biological data in precision medicine. By moving beyond the limitations of classical, single-view random walks, QuVINE leverages quantum-inspired dynamics to capture higher-order topological features and long-range dependencies that are frequently lost in standard diffusion-based models.

### Citation

Please cite the following article if you use QuVINE:

*Quantum-enhanced Network Embeddings via Multi-view Integration for Precision Medicine*, 
A. Bose, F. Utro and L. Parida, 2026. (Under Review)



### Installation

Please follow these steps to run and execute QuVINE. 

```
python -m venv myenv (or your path to the environment)
source myenv/bin/activate
pip install -r requirements.txt
git clone https://github.com/IBM/QuVINE.git
cd QuVINE
pip install -e . 
```

### Usage 

QuVINE supports three kind of walks: random walk with restart (RWR), continuous-time quantum walk (CTQW), and discrete-time quantum walk (DTQW). 
Additionally, QuVINE also fuses the embeddings generated from these walks together to create a fused embedding which is a latent subspace shared by the walk-based embeddings. 

QuVINE supports disease gene prioritization in PPI networks at the moment. It also allows us to hook the embeddings for downstream prediction, classification, or unsupervised learning tasks. 

QuVINE uses [Hydra](https://hydra.cc/) as input format and the code can be run using 

```
python -m quvine.main --config-path configs/ --config-name config.yaml
```

This will create a new directory inside the outputs directory and store the results. 

### Notebook 

To see QuVINE in action refer to the [notebook](https://github.com/IBM/QuVINE/blob/main/notebooks/quvine_embedding.ipynb). 


<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[quvine]: images/quvine_framework.png