from datetime import datetime
import requests
import base64
import json
import time
import os
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from PIL import Image
import io

app = Flask(__name__)
webui_server_url = 'http://127.0.0.1:7860'

# Output directory setup
out_dir = 'api_out'
out_dir_inpaint = os.path.join(out_dir, 'inpaint')
os.makedirs(out_dir_inpaint, exist_ok=True)

# ======= FIXED PARAMETERS =======
FIXED_SETTINGS = {
    "checkpoint_model": "cosplaymix_v42.safetensors",
    "lora_model": "brwnV3-000004.safetensors",
    "lora_strength": 1.3,
    "prompt": "men,wearing,brown,<lora:brwnV3-000004:1.3>,shirt",
    "negative_prompt": "blurred,overlapped,ugly,disfigured,bad quality",
    "seed": 3528166601,
    "steps": 30,
    "sampler_name": "Euler a",
    "width": 576,
    "height": 768,
    "denoising_strength": 0.5,
    "batch_size": 1,
    "cfg_scale": 7.5,
}
# ======= END FIXED PARAMETERS =======

# Function to get timestamp
def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

# Encode image to Base64
def encode_file_to_base64(file):
    try:
        return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding file: {e}")
        return None

# Decode Base64 and save image
def decode_and_save_base64(base64_str, save_path):
    try:
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str))
        return save_path
    except Exception as e:
        print(f"Error saving decoded file: {e}")
        return None

# Resize image while keeping aspect ratio
def resize_image(image_bytes, new_width=256, new_height=384):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert back to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Call Stable Diffusion Inpaint API
def call_inpaint_api(payload):
    url = f"{webui_server_url}/sdapi/v1/img2img"  # ✅ Correct endpoint for inpainting
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if "images" in result:
            save_path = os.path.join(out_dir_inpaint, f"inpaint-{timestamp()}.png")
            decode_and_save_base64(result["images"][0], save_path)
            return save_path
        else:
            print("API response does not contain 'images'")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling inpaint API: {e}")
        return None

LORA_PROMPTS = {
    "brwnV3-000004.safetensors": "men,wearing,brown,<lora:brwnV3-000003:1.3>,shirt",
    "greentV2-000002.safetensors": "men,wearing,green,<lora:greentV2-000002:1.3>,t-shirt",
}

# Flask Route - Upload Image & Mask for Inpainting
@app.route('/inpaint', methods=['POST'])
def inpaint_file():
    if 'file' not in request.files or 'mask' not in request.files:
        return jsonify({"error": "Both image and mask are required"}), 400

    image_file = request.files['file']
    mask_file = request.files['mask']

    if image_file.filename == '' or mask_file.filename == '':
        return jsonify({"error": "No selected file or mask"}), 400

    encoded_image = encode_file_to_base64(image_file)
    encoded_mask = encode_file_to_base64(mask_file)

    if not encoded_image or not encoded_mask:
        return jsonify({"error": "Failed to encode image or mask"}), 500

    # Get the selected LoRA model from the form data
    lora_model = request.form.get("lora_model", FIXED_SETTINGS["lora_model"])

    # Get the corresponding prompt for the selected LoRA
    prompt = LORA_PROMPTS.get(lora_model, FIXED_SETTINGS["prompt"])


    inpaint_payload = {
    "prompt": prompt,  # ✅ Dynamically assigned based on LoRA
    "negative_prompt": FIXED_SETTINGS["negative_prompt"],
    "seed": FIXED_SETTINGS["seed"],
    "steps": FIXED_SETTINGS["steps"],
    "sampler_name": FIXED_SETTINGS["sampler_name"],
    "width": FIXED_SETTINGS["width"],
    "height": FIXED_SETTINGS["height"],
    "batch_size": FIXED_SETTINGS["batch_size"],
    "cfg_scale": FIXED_SETTINGS["cfg_scale"],
    "init_images": [encoded_image],  # Base64 input image
    "mask": encoded_mask,  # Base64 mask image
    "denoising_strength": FIXED_SETTINGS["denoising_strength"],
    "inpainting_fill": 1,
    "inpaint_full_res": True,
    "inpaint_full_res_padding": 32,
    "resize_mode": 1,
    "lora_model": lora_model  # ✅ Include selected LoRA model
    }


    save_path = call_inpaint_api(inpaint_payload)
    if not save_path:
        return jsonify({"error": "Inpainting failed"}), 500

    return jsonify({"message": "Inpainting successful!", "image_url": f"/output/{os.path.basename(save_path)}"})

# Route to serve output images
@app.route('/output/<filename>')
def get_output_image(filename):
    return send_from_directory(out_dir_inpaint, filename)

# Serve the UI
@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Image for Inpainting</title>
    <style>
        canvas { border: 1px solid black; cursor: crosshair; }
        .lora-button {
            border: 2px solid transparent;
            padding: 5px;
            cursor: pointer;
            display: inline-block;
            margin: 5px;
        }
        .lora-button.selected {
            border: 2px solid blue;
        }
    </style>
</head>
<body>
    <h1>Upload an Image & Paint Mask</h1>
    
    <div>
        <img src="static/blue.png" class="lora-button" 
        onclick="selectLora('brwnV3-000003.safetensors', this)" 
        style="width: 50px; height: 50px;">
        <img src="static/red.png" class="lora-button" 
        onclick="selectLora('greentV2-000002.safetensors', this)" 
        style="width: 50px; height: 50px;">
    </div>
    
    <input type="file" id="fileInput">
    <button onclick="clearMask()">Clear Mask</button>
    <button onclick="toggleEraseMode()">Erase Mask</button>
    <button onclick="submitImages()">Submit</button>

    <br><br>
    <canvas id="canvas"></canvas>
    <br>
    <img id="outputImage" style="display:none; width: 300px;">

    <script>
        let canvas = document.getElementById("canvas");
        let ctx = canvas.getContext("2d");
        let painting = false;
        let eraseMode = false;
        let img = new Image();
        let selectedLora = "brwnV3-000004.safetensors";

        // Create an offscreen canvas for the mask
        let maskCanvas = document.createElement("canvas");
        let maskCtx = maskCanvas.getContext("2d");

        function selectLora(lora, element) {
            selectedLora = lora;
            document.querySelectorAll('.lora-button').forEach(btn => btn.classList.remove('selected'));
            element.classList.add('selected');
        }

        document.getElementById("fileInput").addEventListener("change", function(e) {
            let file = e.target.files[0];
            if (!file) return;

            let reader = new FileReader();
            reader.onload = function(event) {
                img.onload = function() {
                    // Set both canvases to match image size
                    canvas.width = img.width;
                    canvas.height = img.height;
                    maskCanvas.width = img.width;
                    maskCanvas.height = img.height;

                    // Draw image on the main canvas
                    ctx.drawImage(img, 0, 0);
                    
                    // Clear the mask canvas
                    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        });

        function toggleEraseMode() {
            eraseMode = !eraseMode;
            document.querySelector("button[onclick='toggleEraseMode()']").innerText = eraseMode ? "Switch to Paint Mask" : "Erase Mask";
        }

        function startPainting(event) {
            painting = true;
            maskCtx.lineWidth = 30;
            maskCtx.lineCap = "round";
            maskCtx.strokeStyle = "white"; // Always paint white for the mask
            maskCtx.globalCompositeOperation = eraseMode ? "destination-out" : "source-over"; // Use "destination-out" to erase
            draw(event);
        }

        function stopPainting() { 
            painting = false; 
            maskCtx.beginPath(); 
            updateCanvas();
        }

        function draw(event) {
            if (!painting) return;
            let x = event.offsetX;
            let y = event.offsetY;

            maskCtx.lineTo(x, y);
            maskCtx.stroke();
            maskCtx.beginPath();
            maskCtx.moveTo(x, y);
            updateCanvas();
        }

        function updateCanvas() {
            ctx.drawImage(img, 0, 0); // Redraw the original image
            ctx.drawImage(maskCanvas, 0, 0); // Apply the mask on top
        }

        function clearMask() { 
            maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            updateCanvas();
        }

        async function submitImages() {
            let formData = new FormData();
            formData.append("file", document.getElementById("fileInput").files[0]);
            formData.append("lora_model", selectedLora); 

            maskCanvas.toBlob(blob => {
                formData.append("mask", blob);
                fetch("/inpaint", { method: "POST", body: formData })
                    .then(res => res.json())
                    .then(data => {
                        if (data.image_url) {
                            document.getElementById("outputImage").src = data.image_url;
                            document.getElementById("outputImage").style.display = "block";
                        }
                    });
            });
        }

        canvas.addEventListener("mousedown", startPainting);
        canvas.addEventListener("mouseup", stopPainting);
        canvas.addEventListener("mousemove", draw);
    </script>
</body>
</html>

    """)

if __name__ == '__main__':
    app.run(debug=True)
