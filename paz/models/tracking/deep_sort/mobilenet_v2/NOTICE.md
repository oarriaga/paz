# MobileNetV2 DeepSORT appearance weights — attribution

`mobilenet_v2_paz_jax.weights.h5` is converted with `port_weights.py` from
`mobilenetv2_bottleneck_wts.pt` (9,084,095 bytes, a PyTorch state dict with
312 entries), the file shipped inside the `deep-sort-realtime` 1.3.2 wheel
(levan92, https://github.com/levan92/deep_sort_realtime, MIT License,
Copyright (c) 2021 Ling Evan). Only the array values are taken from it.

Stated by the package (README, LICENSE and the `MobileNetv2_Embedder`
docstring of the installed 1.3.2 wheel):
- the default embedder is "a pytorch MobilenetV2 (trained on Imagenet)";
- the classification layer is removed and the 1280-d bottleneck is the
  feature;
- the weights file is pushed in the repository and installed with it;
- the tracker is adapted from https://github.com/nwojke/deep_sort, which
  is licensed GPL-3.0.

deep-sort-realtime is used only as an optional test oracle and is never
imported by library code; the paz tracker is written from Wojke et al.
2017 (arXiv:1703.07402).

Inferred, not stated by the package: the network definition and the
state-dict layout (`features.N.conv.M`) match tonylins/pytorch-mobilenet-v2,
so the ImageNet weights most likely come from that project, which is
published under Apache-2.0. TODO: verify the origin and licence of the
weights before release.

No code from deep-sort-realtime, nwojke/deep_sort or tonylins is used. The
paz model and converter are written from Sandler et al. 2018,
"MobileNetV2: Inverted Residuals and Linear Bottlenecks" (arXiv:1801.04381).

## deep-sort-realtime licence

As written in the LICENSE file of the installed
`deep_sort_realtime-1.3.2.dist-info`:

```
MIT License

Copyright (c) [2021] [Ling Evan]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
