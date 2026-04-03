- Protocol A: AdamW primary, learning rate $3 × 10^{−4}$ , weight decay 5 × 10−4 ,cosine annealing, batch size 256, label smoothing 0.1, automatic mixed precision (AMP) with bfloat16 (BF16). Some settings use smaller batches, for example, batch size 128 for ConvNeXt and WRN-50-2, and we label these cases explicitly in the corresponding tables. SmallConvNet: 20 epochs. ResNet-18 and ResNet-50: 50 epochs. Tiny-ImageNet: 20 epochs.

- Protocol B: Stochastic Gradient Descent (SGD), 200 epochs. SGD with momentum 0.9, learning rate 0.1, cosine annealing, batch size 128, no label smoothing, AMP BF16. Weight decay is 10−4 by default; some runs use 5 × 10−4 , and the released protocol strings encode these settings, to extend seed counts where available

- Protocol C: Data-efficient Image Transformer (DeiT)-style ViT training. AdamW, learning rate $5 × 10^{-4}$ , weight decay $5 × 10^{−2}$ , cosine annealing with 30-epoch linear warmup, batch size 128,
label smoothing 0.1, gradient clipping with max norm 1.0, RandAugment with n = 2, m = 9,
Mixup with α = 0.8, CutMix  with α = 1.0, RandomErasing with p = 0.25, 300 epochs.
Follows DeiT recommendations for training ViTs on small datasets.

