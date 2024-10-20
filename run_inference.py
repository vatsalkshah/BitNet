from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import platform
import subprocess
import sys

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    model: str = "models/bitnet_b1_58-3B/ggml-model-i2_s.gguf"
    n_predict: int = 128
    threads: int = 2
    ctx_size: int = 2048
    temperature: float = 0.8

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference(args):
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    command = [
        f'{main_path}',
        '-m', args.model,
        '-n', str(args.n_predict),
        '-t', str(args.threads),
        '-p', args.prompt,
        '-ngl', '0',
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
        "-b", "1"
    ]
    run_command(command)

@app.post("/inference")
async def inference(request: InferenceRequest):
    try:
        run_inference(request)
        return {"status": "success", "message": "Inference completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))