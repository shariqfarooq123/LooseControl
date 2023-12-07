# LooseControl: Lifting ControlNet for Generalized Depth Conditioning
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](#)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![PyTorch](https://img.shields.io/badge/PyTorch_v1.10.1-EE4C2C?&logo=pytorch&logoColor=white) 

This is the official repository for our paper:
>#### [LooseControl: Lifting ControlNet for Generalized Depth Conditioning](#)
> ##### [Shariq Farooq Bhat](https://shariqfarooq123.github.io), [Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/), [Peter Wonka](http://peterwonka.net/) 


[[Project Page]](https://shariqfarooq123.github.io/loose-control/) [[Paper]](https://arxiv.org/abs/2312.03079) [[Demo 🤗]](https://huggingface.co/spaces/shariqfarooq/LooseControl) [[Weights (3D Box Control)]](https://huggingface.co/shariqfarooq/loose-control-3dbox)

![teaser](assets/looseControl_teaser.png)

# Usage
```bash
git clone https://github.com/shariqfarooq123/LooseControl && cd LooseControl
```

Start the UI:
```python
gradio app.py
```

or use via python API:

```python
from loosecontrol import LooseControlNet

lcn = LooseControlNet("shariqfarooq/loose-control-3dbox")

boxy_depth = ...
prompt = "A photo of a snowman in a desert"
negative_prompt = "blurry, text, caption, lowquality,lowresolution, low res, grainy, ugly"


gen_image_1 = lcn(prompt, negative_prompt=negative_prompt, control_image=boxy_depth)

```

Style preserving edits:
```python
# Fix the 'style' and edit
# Edit 'boxy_depth' -> 'boxy_depth_edited'

lcn.set_cf_attention()

gen_image_edited = lcn.edit(boxy_depth, boxy_depth_edited, prompt, negative_prompt=negative_prompt)

```
