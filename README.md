# MIT 6.S191 — Introduction to Deep Learning

This repository documents my study of [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/), using the 2026 lectures, slides, and software labs as the core curriculum.

The objective is not only to complete the course, but to develop working knowledge through active recall, mathematical reasoning, implementation, experimentation, and cumulative review. Where it is genuinely useful, exercises will connect deep learning to dynamics, control, robotics, and physical systems.

## Repository structure

Each lecture has its own directory:

```text
Lecture-01/
Lecture-02/
Lecture-03/
Lecture-04/
Lecture-05/
Lecture-06/
Lecture-07/
Lecture-08/
Lecture-09/
```

A lecture directory may eventually contain:

- A `README.md` with recall, consolidated notes, questions, and review history.
- Focused notebooks or Python scripts.
- Small figures or other supporting material.

Additional directories for the official software labs or larger projects will be introduced only when they are needed.

## Progress

| Lecture | Topic | Video | Notes | Exercises | Review |
|---:|---|:---:|:---:|:---:|:---:|
| 01 | Intro to Deep Learning | ✓ | — | — | — |
| 02 | Deep Sequence Modeling | — | — | — | — |
| 03 | Deep Computer Vision | — | — | — | — |
| 04 | Deep Generative Modeling | — | — | — | — |
| 05 | Deep Reinforcement Learning | — | — | — | — |
| 06 | New Frontiers | — | — | — | — |
| 07 | The Three Laws of AI | — | — | — | — |
| 08 | AI for Science | — | — | — | — |
| 09 | Secrets to Massively Parallel Training | — | — | — | — |

The table records meaningful completion, not mere exposure. For example, “Notes” means that the material has been recalled, checked, and consolidated—not simply copied from the slides.

## Environment

The project uses a deliberately small Conda environment named `deeplearning`. Its direct dependencies are recorded in [`environment.yaml`](environment.yaml); transitive dependencies are intentionally not listed by hand.

Create the environment from scratch with:

```bash
conda env create -f environment.yaml
conda activate deeplearning
```

If the environment already exists, update it with:

```bash
conda env update -n deeplearning -f environment.yaml --prune
conda activate deeplearning
```

Verify the numerical and Apple-silicon setup with:

```bash
python -c "import torch, numpy, matplotlib; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

PyTorch will initially run the small introductory exercises on the CPU. Apple Metal Performance Shaders (`mps`) acceleration will be used when a workload is large enough to benefit from it.

## Working principles

- Prefer understanding and implementation over passive course completion.
- Begin reviews with unaided recall before consulting polished notes.
- Add dependencies only when a concrete exercise requires them.
- Use clear numerical or classical baselines where appropriate.
- Distinguish course-provided material from original notes and implementations.
- Prefer a few well-explained experiments to many unfinished notebooks.

## Attribution

Lecture materials and official laboratory code belong to MIT Introduction to Deep Learning and their respective authors. Any copied or adapted MIT material in this repository must retain its original copyright and attribution notices.

- [MIT 6.S191 course website](https://introtodeeplearning.com/)
- [Official MIT 6.S191 lab repository](https://github.com/MITDeepLearning/introtodeeplearning)

