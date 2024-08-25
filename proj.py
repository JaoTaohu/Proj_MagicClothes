from datetime import datetime
import requests
import base64
import json
import time
import os

webui_server_url = 'http://127.0.0.1:7860'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))

def call_api(api_endpoint, payload):
    url = f'{webui_server_url}/{api_endpoint}'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.reason}")
        print(f"Response content: {response.content}")
        response.raise_for_status()
    
    return response.json()

def call_txt2img_api(payload):
    response = call_api('sdapi/v1/txt2img', payload)
    for index, image in enumerate(response.get('images', [])):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)

def call_img2img_api(payload):
    response = call_api('sdapi/v1/img2img', payload)
    for index, image in enumerate(response.get('images', [])):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)

if __name__ == '__main__':
    # ControlNet parameters
    controlnet_image = encode_file_to_base64(r"C:/Project_MagicClothes/S__18743301.jpg")
    # controlnet_module = "openpose"  # Ensure this matches your ControlNet module
    # controlnet_model = "controlnet11Models_openpose"  # Ensure this matches the actual model file name
    controlnet_module = "depth"  # Ensure this matches your ControlNet module
    controlnet_model = "controlnet11Models_depth"  # Ensure this matches the actual model file name

    # Simplified txt2img payload with ControlNet
    txt2img_payload = {
        "prompt": "masterpiece, (best quality:1.1), 1girl <lora:lora_model:1>",
        "negative_prompt": "",
        "seed": 10,
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M",
        "n_iter": 1,
        "batch_size": 1,
        "alwayson_scripts": {
            "controlnet": {  # Use the correct key for ControlNet
                "args": [
                    {
                        "enabled": True,
                        "image": controlnet_image,
                        "module": controlnet_module,
                        "model": controlnet_model,
                    }
                ]
            }
        }
    }
    print("txt2img_payload:", json.dumps(txt2img_payload, indent=2))  # Log the payload
    call_txt2img_api(txt2img_payload)

    # Simplified img2img payload with ControlNet
    init_images = [
        encode_file_to_base64(r"C:/Project_MagicClothes/S__18743301.jpg"),
    ]
    batch_size = 2
    img2img_payload = {
        "prompt": "masterpiece, (best quality:1.1), 1girl <lora:lora_model:1>",
        "seed": 10,
        "steps": 20,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.5,
        "n_iter": 1,
        "init_images": init_images,
        "batch_size": batch_size if len(init_images) == 1 else len(init_images),
        "alwayson_scripts": {
            "controlnet": {  # Use the correct key for ControlNet
                "args": [
                    {
                        "enabled": True,
                        "image": controlnet_image,
                        "module": controlnet_module,
                        "model": controlnet_model,
                    }
                ]
            }
        }
    }
    print("img2img_payload:", json.dumps(img2img_payload, indent=2))  # Log the payload
    call_img2img_api(img2img_payload)
