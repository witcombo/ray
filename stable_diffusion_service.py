# Updated code to integrate Ray Serve with FastAPI

from io import BytesIO
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
from ray import serve
from ray.serve.handle import DeploymentHandle
from pydantic import BaseModel
import torch
from typing import Optional

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.post(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt_request: PromptRequest, img_size: int = 512):
        prompt = prompt_request.prompt
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)
        return Response(content=image, media_type="image/png")


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
)
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

    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            file_stream = BytesIO()
            image.save(file_stream, "PNG")
            return file_stream.getvalue()


entrypoint = APIIngress.bind(StableDiffusionV2.bind())

if __name__ == "__main__":
    import ray
    import uvicorn

    ray.init(
        runtime_env={
            "pip": [
                "diffusers==0.14.0",
                "transformers==4.25.1",
                "accelerate==0.17.1",
                "fastapi",
                "httpx",
            ]
        }
    )

    handle = serve.run(entrypoint)

    # Run FastAPI application using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
