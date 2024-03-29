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

    async def __call__(self, prompt: str, img_size: int = 512):
        try:
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            file_stream = BytesIO()
            image.save(file_stream, "PNG")
            return Response(content=file_stream.getvalue(), media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# 启动 Ray Serve
serve.start()

# 创建后端和端点
backend = serve.Backend(StableDiffusionV2)
serve.create_endpoint("diffusion_model", backend=backend, route="/diffusion_model", methods=["GET", "POST"])

# 运行服务
serve.run(host="0.0.0.0", port=8888)
