# Neural Network Classifier Similarity Functions Comparison

Compare different similarity functions in neural network classifiers for image classification.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Experiments

**Train a single model:**
```bash
python run_experiments.py --dataset mnist --classifier rbf --latent_dim 2 --mode single
```

**Compare classifiers:**
```bash
python run_experiments.py --dataset mnist --mode compare_classifiers \
    --classifiers dotproduct rbf cosface arcface --latent_dim 10
```

**Compare latent dimensions:**
```bash
python run_experiments.py --dataset mnist --classifier rbf \
    --mode compare_dims --latent_dims 2 5 10 20
```

### Evaluate Results
```bash
python experiments/evaluate.py --checkpoint ./checkpoints/mnist/mnist_rbf_dim2.pt \
    --output_dir ./results/evaluation
```

### Analyze Results
```bash
python analyze_results.py --results_dir ./results/mnist
```

## Available Classifiers

- `dotproduct` - Standard softmax with dot product similarity
- `rbf` - Radial Basis Function kernel
- `cosface` - Cosine similarity with additive margin
- `arcface` - Angular margin-based similarity
- `hybrid` - Combined distance and angular similarity
- `adaptive_rbf` - RBF with learnable bandwidth per class
- `mahalanobis` - Mahalanobis distance-based

## Datasets

- `mnist` - MNIST handwritten digits (28x28)
- `fashion_mnist` - Fashion-MNIST (28x28)
- `cifar10` - CIFAR-10 natural images (32x32)

## Project Structure

```
similarity_functions_project/
├── README.md
├── requirements.txt
├── run_experiments.py          # Main entry point
├── analyze_results.py          # Results analysis
├── models/
│   ├── __init__.py
│   ├── backbones.py           # CNN and ResNet backbones
│   ├── classifiers.py         # 7 similarity functions
│   └── full_model.py          # Complete model
├── utils/
│   ├── __init__.py
│   ├── data.py                # Data loaders
│   ├── metrics.py             # Accuracy, ECE
│   ├── visualization.py       # Plotting
│   └── training.py            # Trainer class
├── experiments/
│   ├── __init__.py
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
└── configs/
    ├── mnist_config.py        # MNIST settings
    └── cifar10_config.py      # CIFAR-10 settings
```

## Outputs

After running experiments, you'll find:

- **Checkpoints**: `./checkpoints/{dataset}/{dataset}_{classifier}_dim{latent_dim}.pt`
- **Visualizations**: `./results/{dataset}/{classifier}_dim{latent_dim}/`
  - 2D latent space plots (for dim=2)
  - t-SNE visualizations (for dim>2)
  - Calibration curves
  - Training curves
- **Analysis**: `./results/{dataset}/`
  - `comparison_summary.json`
  - `results_table.csv`
  - Comparison plots


## Examples

### Week 9-10: Baseline and RBF
```bash
# Compare dot-product vs RBF
python run_experiments.py --dataset mnist --mode compare_classifiers \
    --classifiers dotproduct rbf --latent_dim 10

# Test RBF with 2D latent space
python run_experiments.py --dataset mnist --classifier rbf --latent_dim 2 --mode single
```

### Week 11: All Similarity Functions
```bash
python run_experiments.py --dataset mnist --mode compare_classifiers \
    --classifiers dotproduct rbf cosface arcface hybrid adaptive_rbf mahalanobis \
    --latent_dim 10
```

### Week 12-13: CIFAR-10 Extension
```bash
python run_experiments.py --dataset cifar10 --mode compare_classifiers \
    --classifiers dotproduct rbf arcface hybrid --latent_dim 128
```

## Evaluation Metrics

- **Accuracy**: Classification accuracy on test set
- **ECE**: Expected Calibration Error (lower is better)
- **Latent Space**: Visualization of learned representations
- **Calibration**: Reliability diagrams showing confidence vs accuracy

## Configuration

Modify training parameters in `configs/mnist_config.py` or `configs/cifar10_config.py`:
- Number of epochs
- Learning rate
- Batch size
- Latent dimensions to compare

## Hardware Requirements

- **MNIST/Fashion-MNIST**: CPU or GPU (5-10 min per experiment on CPU)
- **CIFAR-10**: GPU recommended (30-60 min per experiment)

## Citation

If you use this code, please cite the relevant papers:
- RBFSoftmax: [Paper reference]
- ArcFace: [Paper reference]
- CosFace: [Paper reference]
