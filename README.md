# ViT-Classification-CIFAR10  
**Vision Transformer (ViT) Implementation for CIFAR-10 Image Classification**  

![Attention Map Example](attention_map_gif/cifar-index-21.gif)  

---

## 📜 Project Overview  
This repository contains a PyTorch-based, from-scratch implementation of the Vision Transformer (ViT) architecture, applied to the CIFAR-10 image classification task. It was developed as part of the **Honors Deep Learning (IA2)** course during **Semester 6** of the B.Tech curriculum at **KJ SOMAIYA COLLEGE OF ENGINEERING**.  

**Team Members:**  
- Atharva Yewale  
- Joyeeta Basu  
- Aman Jha  

---

## 🚀 Key Features  
- **End-to-End ViT**: Complete implementation of patch embedding, transformer encoder blocks, and classification head.  
- **Configurable Hyperparameters**: Easily adjust patch size, number of layers, hidden dimensions, learning rate, warmup schedule, and more.  
- **Robust Performance**: Achieves between 78% and 82% test accuracy on CIFAR-10, demonstrating the model’s learning capacity on small-scale datasets.  
- **Attention Visualization**: Generate and export animated GIFs that show how the model’s attention shifts across image patches and transformer heads.  
- **Pretrained Checkpoints**: Saved model weights are included in the `model/` directory for immediate inference or fine-tuning.  

---

## 📋 Table of Contents  
1. [Installation](#-installation)  
2. [Usage](#-usage)  
3. [Model Architecture](#-model-architecture)  
4. [Results](#-results)  
5. [Attention Map Visualization](#-attention-map-visualization)  
6. [Project Structure](#-project-structure)  
7. [Further Reading](#-further-reading)  
8. [References](#-references)  
9. [License](#-license)  

---

## 📥 Installation  
1. Clone the repository and navigate into it:  
   ```bash
   git clone https://github.com/nick8592/ViT-Classification-CIFAR10.git
   cd ViT-Classification-CIFAR10
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

---

## ⚙️ Usage  

### Training the Model  
Train the ViT on the CIFAR-10 dataset with customizable settings:  
```bash
python train.py \
  --batch_size 128 \
  --epochs 200 \
  --learning_rate 0.0005 \
  --patch_size 4 \
  --warmup_epochs 10
```  
These options allow you to tailor the training schedule, optimizer behavior, and model granularity to your computational setup.  

### Testing and Inference  
Evaluate the model on the CIFAR-10 test set, or perform inference on individual images:  
```bash
# Full CIFAR-10 test evaluation
env CIFAR=true python test.py --mode cifar

# Single CIFAR-10 image (by index)
python test.py --mode cifar-single --index 5

# Custom image classification
python test.py --mode custom --image_path path/to/your/image.png
```  

### Command-Line Arguments  
#### train.py  
| Argument          | Description                                               | Default  |  
| ----------------- | --------------------------------------------------------- | -------- |  
| `--batch_size`    | Mini-batch size for training                              | 128      |  
| `--num_workers`   | DataLoader worker threads                                 | 2        |  
| `--learning_rate` | Initial learning rate                                     | 5e-4     |  
| `--warmup_epochs` | Epochs with linear learning rate warmup                   | 10       |  
| `--epochs`        | Total number of training epochs                           | 200      |  
| `--device`        | Compute device (`cpu` / `cuda` / `mps`)                    | cuda     |  
| `--image_size`    | Input image resolution                                    | 32       |  
| `--patch_size`    | Side length of each image patch                           | 4        |  
| `--n_classes`     | Number of classification categories                       | 10       |  

#### test.py  
| Argument        | Description                                      | Default                                                                                     |  
| --------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------- |  
| `--mode`        | Inference mode: `cifar`, `cifar-single`, `custom` | `cifar`                                                                                     |  
| `--index`       | Index of the selected CIFAR-10 test image         | 1                                                                                           |  
| `--image_path`  | File path for a custom input image                | None                                                                                        |  
| `--model_path`  | Path to the pretrained ViT checkpoint             | `model/vit-layer12-32-cifar10/ViT_model_199.pt`                                             |  
| `--no_image`    | Disable display of the input image and attention   | False                                                                                       |  

---

## 🏛️ Model Architecture  
The implemented Vision Transformer consists of:  
1. **Patch Embedding Layer**: Splits the input image into fixed-size patches and linearly projects each patch to a vector embedding.  
2. **Positional Encoding**: Adds learnable position embeddings to retain spatial information.  
3. **Transformer Encoder**: A stack of multi-head self-attention layers and feed-forward networks, each followed by layer normalization and residual connections.  
4. **Classification Token & Head**: A dedicated `CLS` token whose final hidden state is passed through a linear layer to produce the class logits.  

Full implementation details are available in [model.py](./model.py).  

---

## 📈 Results  
The following table summarizes the performance of two pretrained ViT variants on CIFAR-10 test data:  

| Pre-trained Model               | Test Accuracy | Test Loss |  
| ------------------------------- | ------------- | --------- |  
| `vit-layer6-32-cifar10`         | 78.31%        | 0.6296    |  
| `vit-layer12-32-cifar10`        | 82.04%        | 0.5560    |  

Checkpoint files for both models are stored under `model/` for quick loading and further fine-tuning.  

---

## 🎞️ Attention Map Visualization  
Understanding where the ViT focuses can offer valuable interpretability. This repository provides animated GIFs that visualize the self-attention weights across different heads and layers.  

- **Example GIF 1**  
  ![Attention Map Example 1](attention_map_gif/cifar-index-10.gif)  

- **Example GIF 2**  
  ![Attention Map Example 2](attention_map_gif/cifar-index-13.gif)  

To generate your own attention visualizations, run the `visualize_attention_map.ipynb` notebook.  

---

## 📁 Project Structure  
```bash
ViT-Classification-CIFAR10/
├── data/                          # Dataset storage and preprocessing scripts
├── attention_map_gif/             # Generated attention GIFs
├── model/                         # Pretrained ViT checkpoints
│   ├── vit-layer6-32-cifar10/
│   └── vit-layer12-32-cifar10/
├── output/                        # Training and evaluation logs
├── LICENSE                        # MIT License terms
├── README.md                      # Project overview and instructions
├── requirements.txt               # Python dependencies
├── visualize_attention_map.ipynb  # Notebook for attention map generation
├── model.py                       # ViT model implementation
├── train.py                       # Training script
└── test.py                        # Testing and inference script
```  

---

## 📖 Further Reading  
For an in-depth exploration of Vision Transformers and attention mechanisms, see:  
- **Understanding Vision Transformers: A Game Changer in Computer Vision**  
- **Self-Attention vs. Cross-Attention in Computer Vision**  
- **Attention in Computer Vision: Revolutionizing How Machines “See” Images**  

All articles are available on the [Medium profile](https://medium.com/@weichenpai).  

---

## 📚 References  
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv preprint arXiv:2010.11929.  
- Karpathy’s [Step-by-Step Guide to Image Classification with Vision Transformers (ViT)](https://comsci.blog/posts/vit).  
- Pulfer, B. "Vision Transformers from Scratch (PyTorch): A step-by-step guide." Medium.  
- Chhajed, S. "PyTorch-Scratch-Vision-Transformer-ViT". GitHub.  
- Jacobgil. "Exploring Explainability for Vision Transformers."  

---

## ⚖️ License  
This project is released under the MIT License. See the [LICENSE](LICENSE) file for full details.
