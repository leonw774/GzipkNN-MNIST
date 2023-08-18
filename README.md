# MNIST Digit Recognition with Lossless Image Compression Formats

This little project is inspired by "“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors" (https://aclanthology.org/2023.findings-acl.426/)

## Parameters

- Two lossless image compression format:
  - PNG
  - QOI

- Use grayscale or binary image
  - Grayscale: original
  - Binary: if pixel value > 0, then it become 255

- Using kNN with $k = 2, 8, 32$

- Concatenate two images horizontally or vertically

## Results

See `cm/` for confusion matrices

### Using PNG format

PNG-Grayscale| k=2 | k=8 | k=32
-------------|-----|-----|-----
concat: hori |32.8%|38.3%|41.8%
concat: vert |30.6%|33.1%|33.9%

PNG-Binary   | k=2 | k=8 | k=32
-------------|-----|-----|-----
concat: hori |14.9%|17.7%|19.2%
concat: vert |10.3%|14.5%|16.7%


### Using QOI format

QOI-Grayscale| k=2 | k=8 | k=32
-------------|-----|-----|-----
concat: hori |33.7%|36.9%|39.0%
concat: vert |47.0%|54.6%|58.5%

QOI-Binary   | k=2 | k=8 | k=32
-------------|-----|-----|-----
concat: hori |59.6%|76.8%|67.3%
concat: vert |82.9%|69.1%|83.6%
