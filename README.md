# ViT-Classification-CIFAR10  
**Vision Transformer Implementation for CIFAR-10 Classification**  

![Attention Map Example](attention_map_gif/cifar-index-21.gif)  

---

## 📜 Project Overview  
This project was developed as part of the **Honor Deep Learning (IA2)** course during **Semester 6** of the B.Tech program at **KJ SOMAIYA OLLEGE OF ENGINEERING**.  
**Collaborators**:  
- **Atharva Yewale**  
- **Joyeeta Basu**  
- **Aman Jha**  

---

## 🚀 Features  
- **Scratch Implementation** of Vision Transformer (ViT) using PyTorch.  
- Training/evaluation on **CIFAR-10 dataset** (78-82% test accuracy).  
- Customizable hyperparameters (patch size, learning rate, warmup epochs).  
- Attention map visualization via animated GIFs.  
- Pre-trained models hosted on Hugging Face Hub.  

---

## 📋 Table of Contents  
1. [Installation](#-installation)  
2. [Usage](#-usage)  
3. [Model Architecture](#-model-architecture)  
4. [Results](#-results)  
5. [Visualization](#-attention-map-visualization)  
6. [References](#-references)  

---

## 📥 Installation  
```bash
git clone https://github.com/nick8592/ViT-Classification-CIFAR10.git
cd ViT-Classification-CIFAR10
pip install -r requirements.txt

## Usage

To train the Vision Transformer on the CIFAR-10 dataset, you can run the following command:

```bash
python train.py --batch_size 128 --epochs 200 --learning_rate 0.0005
```

To test the Vision Transformer on the CIFAR-10 dataset, single CIFAR image or custom single image, you can run the following command:

```bash
(CIFAR)        python test.py --mode cifar
(CIFAR single) python test.py --mode cifar-single --index 5
(custom)       python test.py --mode custom --image_path <path_to_image>
```

## Arguments

The following arguments can be passed to the `train.py` script:

- `--batch_size`: Batch size for training (default: 128)
- `--num_workers`: Number of workers for data loading (default: 2)
- `--learning_rate`: Initial learning rate (default: 5e-4)
- `--warmup_epochs`: Number of warmup epochs for learning rate (default: 10)
- `--epochs`: Total number of training epochs (default: 200)
- `--device`: Device to use for training, either "cpu", "cuda", or "mps" (default: "cuda")
- `--image_size`: Size of the input image (default: 32)
- `--patch_size`: Size of the patches to divide the image into (default: 4)
- `--n_classes`: Number of output classes (default: 10)

For a full list of arguments, refer to the [train.py](./train.py) file.

The additional arguments can be passed to the `test.py` script:

- `--mode`: Type of testing mode (default: cifar)
- `--index`: Index of choosen image within the batches (default: 1)
- `--image_path`: Path of custom image (default: None)
- `--model_path`: Path of ViT model (default: model/vit-classification-cifar10-colab-t4/ViT_model_199.pt)
- `--no_image`: Option of disable showing image (default: False)

For a full list of arguments, refer to the [test.py](./test.py) file.

## Model Architecture

The Vision Transformer model implemented in this repository consists of the following key components:

- **Embedding Layer**: Converts image patches into vector embeddings.
- **Transformer Encoder**: Processes embeddings with self-attention and feedforward layers.
- **Classification Head**: A token added to the sequence for final classification.

For details, check the implementation in [model.py](./model.py).

## Results

|   Pre-trained Model    | Test Accuracy | Test Loss |                       Hugging Face Link                       |
| :--------------------: | :-----------: | :-------: | :-----------------------------------------------------------: |
| vit-layer6-32-cifar10  |    78.31%     |  0.6296   | [link](https://huggingface.co/nickpai/vit-layer6-32-cifar10)  |
| vit-layer12-32-cifar10 |    82.04%     |  0.5560   | [link](https://huggingface.co/nickpai/vit-layer12-32-cifar10) |

```bash
./ViT-Classification-CIFAR10
├── data
├── attention_map_gif
├── model
│   ├── vit-layer6-32-cifar10
│   │   └── vit-layer6-32-cifar10-199.pt
│   └── vit-layer12-32-cifar10
│       └── vit-layer12-32-cifar10-199.pt
├── output
│   ├── vit-layer6-32-cifar10
│   └── vit-layer12-32-cifar10
├── LICENSE
├── README.md
├── requirements.txt
├── visualize_atteniton_map.ipynb
├── model.py
├── test.py
└── train.py
```

## Attention Map Visualization

To better understand how the Vision Transformer (ViT) attends to different regions of the input image during classification, this repository includes example attention maps that show the model’s focus across transformer layers. The attention maps can be visualized as GIFs, which dynamically highlight how attention shifts through the layers and heads.

Below are two example GIFs, showcasing attention maps for different images from the CIFAR-10 dataset:

### Example 1

![Attention Map Example 1](attention_map_gif/cifar-index-10.gif)

### Example 2

![Attention Map Example 2](attention_map_gif/cifar-index-13.gif)

The GIFs demonstrate how ViT processes each patch in the image, showing which areas are more influential in the final classification. To create similar visualizations, use the `visualize_attention_map.ipynb` notebook provided in the repository.

## Further Reading

For a deeper understanding of Vision Transformers and their applications in computer vision, check out my articles on Medium:

- **[Understanding Vision Transformers: A Game Changer in Computer Vision](https://medium.com/@weichenpai/understanding-vision-transformers-a-game-changer-in-computer-vision-dd40980eb750)**  
- **[Self-Attention vs. Cross-Attention in Computer Vision](https://medium.com/@weichenpai/self-attention-vs-cross-attention-in-computer-vision-4623b6d4706f)**
- **[Attention in Computer Vision: Revolutionizing How Machines “See” Images](https://medium.com/@weichenpai/attention-in-computer-vision-revolutionizing-how-machines-see-images-8bef2f1fc986)**   


## References

This implementation is inspired by the Vision Transformer paper and other open-source implementations:

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch-Scratch-Vision-Transformer-ViT](https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT)
- [Step-by-Step Guide to Image Classification with Vision Transformers (ViT)](https://comsci.blog/posts/vit)
- [Vision Transformers from Scratch (PyTorch): A step-by-step guide](https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- [Exploring Explainability for Vision Transformers](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
