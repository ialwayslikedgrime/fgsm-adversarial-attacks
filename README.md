# Adversarial Attacks on Neural Networks — FGSM on MNIST

Implementation of the **Fast Gradient Sign Method (FGSM)** from scratch in PyTorch, applied to a LeNet-style classifier trained on MNIST. Based on the seminal paper by Goodfellow, Shlens & Szegedy (ICLR 2015).

---

## What this covers

This notebook is a complete, hands-on introduction to adversarial attacks on image classifiers:

- **Theory** — mathematical derivation of FGSM from first-order optimization
- **Implementation** — FGSM and targeted FGSM from scratch in PyTorch
- **Evaluation** — attack success rate, robust accuracy, perturbation norms, confidence degradation
- **Visualization** — interactive explorer for adversarial examples (epsilon slider, sample selector)
- **Targeted attacks** — forcing misclassification toward a specific class

---

## Repo structure

```
fgsm-adversarial-attacks/
├── adversarial.ipynb          # Main notebook
├── model/
│   └── lenet_mnist_model.pth  # Pretrained LeNet weights
├── materials/
│   ├── FGSM_Paper.pdf         # Goodfellow et al. (2015)
│   ├── intro_2026.pdf         # Course introduction slides
│   └── ml_report_adversarial.pdf
├── requirements.txt
└── README.md
```

> **Note:** The `data/MNIST/` folder is not included — it downloads automatically via `torchvision` on first run.

---

## Getting started

```bash
git clone https://github.com/ialwayslikedgrime/fgsm-adversarial-attacks.git
cd fgsm-adversarial-attacks
pip install -r requirements.txt
jupyter notebook adversarial.ipynb
```

The notebook loads the pretrained model from `model/lenet_mnist_model.pth` and runs immediately — no training required.

---

## Key results

| Epsilon (ε) | Robust Accuracy | Attack Success Rate |
|-------------|-----------------|---------------------|
| 0.00        | ~99%            | ~1%                 |
| 0.05        | ~95%            | ~5%                 |
| 0.10        | ~87%            | ~13%                |
| 0.20        | ~40%            | ~60%                |
| 0.30        | ~6%             | ~94%                |

The model collapses rapidly between ε=0.1 and ε=0.3 — the practically dangerous range where perturbations remain imperceptible to humans.

---

## Architecture

**LeNet-style CNN** trained on MNIST:

```
Input (1×28×28)
→ Conv2d(1, 32, 3) + ReLU
→ Conv2d(32, 64, 3) + ReLU + MaxPool2d(2) + Dropout(0.25)
→ Flatten → Linear(9216, 128) + ReLU + Dropout(0.5)
→ Linear(128, 10) → LogSoftmax
```

Baseline accuracy: **~99%** on clean MNIST test set.

---

## FGSM — core formula

```
x_adv = clip( x + ε · sign(∇ₓ L(θ, x, y)), x_min, x_max )
```

Where `∇ₓ L` is the gradient of the cross-entropy loss with respect to the input image. A single forward-backward pass is sufficient.

For **targeted** attacks, the sign is flipped to minimize loss toward the target class:

```
x_adv = clip( x − ε · sign(∇ₓ L(θ, x, y_target)), x_min, x_max )
```

---

## Reference

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and harnessing adversarial examples.* International Conference on Learning Representations (ICLR 2015). https://arxiv.org/abs/1412.6572