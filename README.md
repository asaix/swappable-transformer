# Transformer Implementation

This is a small part of the code I used for an ablation study I did on transformer architecture. It contains the methods for implementations of different parts of the standard transformer architecture (see #methods).

## About

- Modular and easy to import into another project or change methods
- Parameter choices and implementation based on the Attention Is All You Need paper by Vaswani et Al. (**a few things are different, see below**)
- Implemented using Pytorch


## Methods

- Attention
  - MHA 
  - GQA 
- Normalization
  - LayerNorm
  - RMSNorm
- Positional Encoding
  - SinusoidalPE
  - RoPE
- Transformer stack
  - EncoderLayer
  - DecoderLayer
  - Encoder
  - Decoder
  - Transformer

## Deviations from AIAYN paper
- Use pre-norm instead of post-norm  
- No shared weight matrix across both embedding layers and the pre-softmax projection
- Separate source and target vocabularies
- Bias terms added to the linear projections and weight matrices

## AI Usage

0%. I made this to learn about how transformers are implemented, so there wouldn't have been any point to using AI to write the code.

## References

1. Vaswani et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Zhang et al. (2019). *Root Mean Square Layer Normalization*. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
3. Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
4. Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

## License

This code is licensed under the Unlicense. 

[Full copy here](LICENSE).
