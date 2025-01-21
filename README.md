# Thanos - Method of Pruning LLMs
Official PyTorch implementation of **Thanos**, as presented in our paper:

**Thanos: A Block-wise Pruning Algorithm for Efficient Large Language Model
Compression** </br>
*Ivan Ilin*<br>
GenAI Center of Excellence, King Abdullah University of Science and Technology, Thuwal, Saudi Arabia<br>
[Paper](link)

```bibtex
@article{ilin2024thanos,
  title={The name of the paper will be here}, 
  author={Ivan Ilin, Peter Richtarik},
  year={2024},
  journal={arXiv preprint arXiv:}
}
```

---

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
The [scripts](scripts) directory contains all the bash commands to replicate the main results.

Below is an example command for pruning LLaMA-7B with Thanos, to achieve unstructured 50% sparsity.
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method thanos \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/thanos/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented four pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`, `thanos`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method thanos \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/thanos/ 
```

### Pruning LLaMA-2
For [LLaMA-2](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-2-7b-hf` (take `7b` as an example):
```sh 
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method thanos \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/thanos/
```

## Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) and [SparceGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.