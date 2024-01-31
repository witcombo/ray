from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import torch

from ray import serve

app = FastAPI()

@serve.deployment(name="diffusion_model", num_replicas=1)
class StableDiffusionV2:
    def __init__(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")

    async def __call__(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image

@serve.deployment(name="api_ingress")
class APIIngress:
    async def __call__(self, prompt: str, img_size: int = 512):
        try:
            handle = serve.get_handle("diffusion_model")
            image = await handle.remote(prompt, img_size=img_size)
            file_stream = BytesIO()
            image.save(file_stream, "PNG")
            return Response(content=file_stream.getvalue(), media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

entrypoint = APIIngress.deploy()
serve.start(http_options={"host": "0.0.0.0", "port": 8888})
