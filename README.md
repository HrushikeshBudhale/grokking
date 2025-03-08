# grokking
Reimplementation of paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) for experimenting with grokking phenomenon on small algorithmic dataset.  

<p align="center">
  <img width="400" src="https://github.com/HrushikeshBudhale/grokking/blob/main/results/GrokFast_Accuracy_graph.png?raw=true" alt="Accuracy Graph">
  <img width="400" src="https://github.com/HrushikeshBudhale/grokking/blob/main/results/GrokFast_Loss_graph.png?raw=true" alt="Loss Graph">
</p>

## Installation

1. Create a conda environment with python 3.10.

    ```bash
    conda create -n grok python=3.10
    conda activate grok
    ```

2. Install the dependencies.

    ```bash
    pip install -r requirements.txt
    ```

3. Run the code.

    ```bash
    python train.py
    ```

## References
Papers:
* [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
* [Grokfast: Accelerated Grokking by Amplifying Slow Gradients](arxiv.org/abs/2405.20233)

Code:
* [OpenAI/grok](https://github.com/openai/grok)
* [ironjr/grokfast](https://github.com/ironjr/grokfast)
* [danielmamay/grokking](https://github.com/danielmamay/grokking)
