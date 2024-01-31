from io import BytesIO
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
import torch

from ray import serve

app = FastAPI()

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

    async def __call__(self, request):
        prompt = request.query_params.get("prompt", "")
        img_size = int(request.query_params.get("img_size", 512))

        try:
            image = await self.pipe.remote(prompt, img_size=img_size)
            file_stream = BytesIO()
            image.save(file_stream, "PNG")
            return Response(content=file_stream.getvalue(), media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

serve.start()
serve.create_backend("diffusion_model", StableDiffusionV2)
serve.create_endpoint("diffusion_model", backend="diffusion_model", route="/diffusion_model")

# Use `block=False` to allow the script to run continuously.
serve.run(host="0.0.0.0", port=8888, block=False)
