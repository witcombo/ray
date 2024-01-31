# __example_code_start__

from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import json
import requests
import os
import random
import string

from ray import serve
from ray.serve.handle import DeploymentHandle


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

        # Generate a random filename for the image
        filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".png"

        # Save the image with the random filename
        file_path = os.path.join("images", filename)
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        with open(file_path, "wb") as f:
            f.write(file_stream.getvalue())

        # Prepare headers and body for image upload
        headers = {
            "Authorization": "tvIlCBD1cmkrQWafhoU3Gi7gb4KSdRuP",
        }
        files = {
            'smfile': (filename, file_stream.getvalue(), 'image/png'),
            'format': (None, 'json'),
        }

        # Upload the image to the specified image hosting service
        upload_url = "https://sm.ms/api/v2/upload"
        response_upload = requests.post(upload_url, headers=headers, files=files)

        # Check if the upload was successful
        if response_upload.status_code == 200:
            print("Image uploaded successfully.")
            print(f"Image URL: {response_upload.json().get('data').get('url')}")
        else:
            print(f"Failed to upload image. Status code: {response_upload.status_code}")

        # Return the image as a response
        return Response(content=file_stream.getvalue(), media_type="image/png")


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

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image


entrypoint = APIIngress.bind(StableDiffusionV2.bind())

# __example_code_end__


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
