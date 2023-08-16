# MNIST Digit Recognition with Lossless Image Compression Formats

This little project is inspired by "“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors" (https://aclanthology.org/2023.findings-acl.426/)

## Parameters

- Two lossless image compression format:
  - PNG
  - QOI

- Using kNN with $k = 2, 8, 32$

- Concatenate two images horizontally or vertically

## Results

See `cm/` for confusion matrices

### Using PNG format

PNG          | k=2 | k=8 | k=32
-------------|-----|-----|------
concat: hori |32.8%|38.3%|41.8%
concat: vert |30.6%|33.1%|33.9%


### Using QOI format

QOI          | k=2 | k=8 | k=32
-------------|-----|-----|------
concat: hori |33.7%|36.9%|39.0%
concat: vert |47.0%|54.6%|58.5%


- On image concatenation:
  - For PNG format, horizontal concatenation is better
  - For QOI format, vertically concatenation is better

- On value of k for kNN:
  - Larger k gives better the accuracy

- Overall, using QOI than using PNG, setting larger k, and concatenate image vertically would have better result.