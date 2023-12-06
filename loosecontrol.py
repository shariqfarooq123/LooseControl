from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
import torch
import PIL
import PIL.Image
from diffusers.loaders import UNet2DConditionLoadersMixin
from typing import Dict
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
import functools
from cross_frame_attention import CrossFrameAttnProcessor

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"
NEGATIVE_PROMPT = "blurry, text, caption, lowquality, lowresolution, low res, grainy, ugly"

def attach_loaders_mixin(model):
    # hacky way to make ControlNet work with LoRA. This may not be required in future versions of diffusers.
    model.text_encoder_name = TEXT_ENCODER_NAME
    model.unet_name = UNET_NAME
    r"""
    Attach the [`UNet2DConditionLoadersMixin`] to a model. This will add the
    all the methods from the mixin 'UNet2DConditionLoadersMixin' to the model.
    """
    # mixin_instance = UNet2DConditionLoadersMixin()
    for attr_name, attr_value in vars(UNet2DConditionLoadersMixin).items():
        # print(attr_name)
        if callable(attr_value):
            # setattr(model, attr_name, functools.partialmethod(attr_value, model).__get__(model, model.__class__))
            setattr(model, attr_name, functools.partial(attr_value, model))
    return model

def set_attn_processor(module, processor, _remove_lora=False):
    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            if not isinstance(processor, dict):
                module.set_processor(processor, _remove_lora=_remove_lora)
            else:
                module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in module.named_children():
        fn_recursive_attn_processor(name, module, processor)



class ControlNetX(ControlNetModel, UNet2DConditionLoadersMixin):
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    # This may not be required in future versions of diffusers.
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

class ControlNetPipeline:
    def __init__(self, checkpoint="lllyasviel/control_v11f1p_sd15_depth", sd_checkpoint="runwayml/stable-diffusion-v1-5") -> None:
        controlnet = ControlNetX.from_pretrained(checkpoint)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        sd_checkpoint, controlnet=controlnet, requires_safety_checker=False, safety_checker=None,
                        torch_dtype=torch.float16)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    @torch.no_grad()
    def __call__(self, 
                    prompt: str="",
                    height=512,
                    width=512, 
                    control_image=None, 
                    controlnet_conditioning_scale=1.0, 
                    num_inference_steps: int=20,
                   **kwargs) -> PIL.Image.Image:
        
        out =  self.pipe(prompt, control_image,
                            height=height, width=width,
                            num_inference_steps=num_inference_steps,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            **kwargs).images

        return out[0] if len(out) == 1 else out
    
    def to(self, *args, **kwargs):
        self.pipe.to(*args, **kwargs)
        return self


class LooseControlNet(ControlNetPipeline):
    def __init__(self, loose_control_weights="shariqfarooq/loose-control-3dbox", cn_checkpoint="lllyasviel/control_v11f1p_sd15_depth", sd_checkpoint="runwayml/stable-diffusion-v1-5") -> None:
        super().__init__(cn_checkpoint, sd_checkpoint)
        self.pipe.controlnet = attach_loaders_mixin(self.pipe.controlnet)
        self.pipe.controlnet.load_attn_procs(loose_control_weights)

    def set_normal_attention(self):
        self.pipe.unet.set_attn_processor(AttnProcessor())

    def set_cf_attention(self, _remove_lora=False):
        for upblocks in self.pipe.unet.up_blocks[-2:]:
            set_attn_processor(upblocks, CrossFrameAttnProcessor(), _remove_lora=_remove_lora)

    def edit(self, depth, depth_edit, prompt, prompt_edit=None, seed=42, seed_edit=None, negative_prompt=NEGATIVE_PROMPT, controlnet_conditioning_scale=1.0, num_inference_steps=20, **kwargs):
        if prompt_edit is None:
            prompt_edit = prompt

        if seed_edit is None:
            seed_edit = seed
    
        seed = int(seed)
        seed_edit = int(seed_edit)
        control_image = [depth, depth_edit]
        prompt = [prompt, prompt_edit]
        generator = [torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed_edit)]
        gen = self.pipe(prompt, control_image=control_image, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator, num_inference_steps=num_inference_steps, negative_prompt=negative_prompt, **kwargs)[-1]
        return gen