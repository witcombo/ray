#!/home/ray/anaconda3/bin/python

model_id = "stabilityai/stable-diffusion-2-1"
prompt = "a photo of an astronaut riding a horse on mars"

import ray

ray.init(
    runtime_env={
        "pip": [
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "diffusers>=0.13.1",
            "xformers>=0.0.16",
            "torch<2",
        ]
    }
)

import ray.data
import pandas as pd

ds = ray.data.from_pandas(pd.DataFrame([prompt] * 4, columns=["prompt"]))

class PredictCallable:
    def __init__(self, model_id: str, revision: str = None):
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        # Use xformers for better memory usage
        from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
        import torch

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_xformers_memory_efficient_attention(
            attention_op=MemoryEfficientAttentionFlashAttentionOp
        )
        # Workaround for not accepting attention shape using VAE for Flash Attention
        self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        self.pipe = self.pipe.to("cuda")

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        import torch
        import numpy as np

        # Set a different seed for every image in batch
        self.pipe.generator = [
            torch.Generator(device="cuda").manual_seed(i) for i in range(len(batch))
        ]
        images = self.pipe(list(batch["prompt"])).images
        return {"images": np.array(images, dtype=object)}



preds = ds.map_batches(
    PredictCallable,
    batch_size=1,
    fn_constructor_kwargs=dict(model_id=model_id),
    concurrency=1,
    batch_format="pandas",
    num_gpus=1,
)
results = preds.take_all()
