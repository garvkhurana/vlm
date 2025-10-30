import os
import cv2
import json
import torch
import fitz
import tempfile
import shutil
import re
from PIL import Image
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForImageTextToText
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {DEVICE}")
MODEL_ID = "LiquidAI/LFM2-VL-450M"
FRAME_RATE = 1

# Load models globally
print(" Loading vision-language model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(" Model loaded successfully.")

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in environment vars.")
client = Groq(api_key=GROQ_API_KEY)

# Flask setup
app = Flask(__name__)
CORS(app)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file, fallback to OCR if standard extraction fails."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    text = text.strip()
    print("Extracted SOP text (first 200 chars):", text[:200])
    # Fallback to OCR if text is too short (likely scanned PDF)
    if not text or len(text) < 50:
        print("No/insufficient text extracted. Trying OCR fallback...")
        import pytesseract
        import numpy as np
        from PIL import Image
        ocr_text = ""
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                # Render page to an image
                pix = doc[page_num].get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                arr = np.array(img)
                # Run OCR (assume English)
                page_text = pytesseract.image_to_string(arr, lang="eng")
                ocr_text += page_text + "\n"
        print("OCR Fallback Extracted (first 200 chars):", ocr_text[:200])
        return ocr_text.strip()
    return text

def extract_sop_points(sop_text):
    """Split SOP text into individual checklist points organized by steps."""
    lines = [line.strip() for line in sop_text.split('\n') if line.strip()]
    points = []
    current_step = "General"
    current_point = ""
    
    for line in lines:
        # Skip title, purpose, scope, etc. - focus on procedure steps
        if any(keyword in line.lower() for keyword in ['title:', 'purpose', 'scope', 'responsibilities', 'definitions', 'references', 'revision']):
            continue
            
        # Check for main section headers (A. Material Preparation, B. Setting Up, etc.)
        # Pattern: Letter followed by period and text, OR "Step" followed by number/text
        if re.match(r'^[A-Z]\.\s+[A-Z]', line) or re.match(r'^(?:Step|Section|Phase)\s+\d+', line, re.IGNORECASE):
            # Save previous point if exists
            if current_point:
                points.append({
                    'step': current_step,
                    'point': current_point.strip()
                })
                current_point = ""
            
            # Set new step
            current_step = line.strip()
            continue
        
        # Check for bullet points or numbered items
        # Matches: ● bullets, numbered items, lettered items, dashes, etc.
        if re.match(r'^[●○▪►•\-\*]\s+', line) or re.match(r'^\d+[\.)]\s+', line) or re.match(r'^[a-z][\.)]\s+', line):
            # Save previous point if exists
            if current_point:
                points.append({
                    'step': current_step,
                    'point': current_point.strip()
                })
            
            # Start new point - remove bullet/number prefix
            current_point = re.sub(r'^[●○▪►•\-\*\d]+[\.)]*\s+', '', line).strip()
        else:
            # Continue current point (multi-line content)
            if current_point:
                current_point += " " + line
            elif current_step != "General":  # Start new point only if we're in a step
                current_point = line
    
    # Add final point if exists
    if current_point:
        points.append({
            'step': current_step,
            'point': current_point.strip()
        })
    
    # Filter out empty or very short points
    points = [p for p in points if len(p['point']) > 10]
    
    # Add debug print for points
    print("First 5 extracted SOP points:", points[:5])
    return points

def extract_frames(video_file):
    """Extract frames from video file object to memory."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
    
    try:
        # Save video temporarily
        video_file.save(temp_video_path)
        
        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        interval = int(fps * FRAME_RATE)
        frames = []
        frame_count = 0

        print(f"📹 Extracting frames (FPS: {fps})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                # Convert frame to PIL Image and store in memory
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frames.append(pil_image)
            frame_count += 1
            
        cap.release()
        print(f" Extracted {len(frames)} frames")
        return frames
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def caption_frames(frames):
    """Generate captions for frames."""
    inspection_prompt = "Describe in detail what is happening in this manufacturing process. Focus on worker actions, equipment usage, safety measures, and procedural steps. If unclear, say 'Unclear content'."
    captions = []
    
    for i, image in enumerate(tqdm(frames, desc="🎬 Analyzing frames")):
        try:
            start_sec = i
            end_sec = i + 1
            start_time = f"{int(start_sec//60):02}:{int(start_sec%60):02}"
            end_time = f"{int(end_sec//60):02}:{int(end_sec%60):02}"

            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": inspection_prompt},
                ],
            }]

            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)

            caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            captions.append({
                "frame": i,
                "start_time": start_time,
                "end_time": end_time,
                "caption": caption.strip()
            })
        except Exception as e:
            print(f" Error processing frame {i}: {e}")
            captions.append({
                "frame": i,
                "start_time": start_time,
                "end_time": end_time,
                "caption": "Error processing frame"
            })
    
    return captions

def generate_analysis_report(captions_json, sop_points):
    """Generate analysis report using Groq."""
    # Group points by step for better organization
    steps = {}
    for point in sop_points:
        if point['step'] not in steps:
            steps[point['step']] = []
        steps[point['step']].append(point['point'])
    
    prompt = f"""
You are a professional manufacturing process auditor with expertise in quality control and safety compliance.

Below is JSON data describing observations from a manufacturing video:
{json.dumps(captions_json, indent=2)}

And below is the Standard Operating Procedure (SOP) organized by steps:
{json.dumps(steps, indent=2)}

Generate a **comprehensive manufacturing inspection report** that:

1. **Executive Summary** - Brief overview of the process and key findings
2. **Step-by-Step Analysis** - Detailed analysis of each step and its requirements
3. **Compliance Review** - Which procedures were followed and which were missed
4. **Safety Assessment** - Identification of safety concerns and violations
5. **Quality Control Findings** - Process quality and consistency issues
6. **Recommendations** - Specific improvements for each step
7. **Conclusion** - Overall assessment and priority actions

Requirements:
- Analyze each step separately
- Use professional, technical language
- Be specific with timestamps when referencing observations
- Clearly mark compliance gaps
- Format in clean Markdown with proper headers
"""

    print("Generating analysis report...")
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a senior manufacturing quality auditor and safety expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3000,
    )

    return response.choices[0].message.content

def generate_compliance_checklist(captions_json, sop_points):
    """Generate structured compliance checklist using Groq."""
    checklist = []
    
    print(f"🔍 Analyzing {len(sop_points)} SOP points...")
    
    # Group points by step
    steps = {}
    for point in sop_points:
        step = point['step']
        if step not in steps:
            steps[step] = []
        steps[step].append(point['point'])
    
    # Process each step
    for step, points in steps.items():
        step_prompt = f"""
Analyze these SOP requirements for {step}:

Requirements:
{json.dumps(points, indent=2)}

Video Observations:
{json.dumps(captions_json, indent=2)}

Return ONLY a valid JSON array. Each object must have this EXACT format:
[
    {{
        "step": "{step}",
        "point": "the original requirement text",
        "satisfied": true,
        "evidence": "specific evidence from observations",
        "timestamps": ["00:00", "00:05"]
    }}
]

Rules:
- Return ONLY the JSON array, no markdown, no explanation
- Be strict: only mark satisfied=true if there is clear evidence
- If no evidence, set satisfied=false
- Always include all fields
"""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a strict compliance auditor. Return ONLY valid JSON array."},
                    {"role": "user", "content": step_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            result_raw = response.choices[0].message.content.strip()
            print("----------------\nRAW JSON returned from Groq for step", step, ":\n", result_raw[:500])
            # Clean markdown formatting if present
            if result_raw.startswith("```"):
                result_raw = result_raw.split("```",1)[1]
                if result_raw.startswith("json"):
                    result_raw = result_raw[4:]
                result_raw = result_raw.strip()
            try:
                result = json.loads(result_raw)
                checklist.extend(result)
                print(f" ✓ Analyzed step: {step}")
            except json.JSONDecodeError:
                print(f" ✗ Error parsing response for step: {step}")
                checklist.append({
                    "step": step,
                    "point": "Error analyzing this step",
                    "satisfied": False,
                    "evidence": "Error parsing response",
                    "timestamps": []
                })
        except Exception as e:
            print(f" ✗ Error processing step: {str(e)}")
            checklist.append({
                "step": step,
                "point": "Error processing this step",
                "satisfied": False,
                "evidence": "Error processing this requirement",
                "timestamps": []
            })
    print(f" Completed analysis of {len(checklist)} points")
    return checklist

@app.route("/")
def index():
    """Serve main page."""
    return render_template("index2.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint."""
    temp_dir = None
    try:
        # Get uploaded files
        video_file = request.files.get("video")
        sop_file = request.files.get("sop")

        if not video_file or not sop_file:
            return jsonify({"error": "Both video and SOP files are required."}), 400

        print(f" Processing files: {video_file.filename}, {sop_file.filename}")

        # Create temporary directory for SOP
        temp_dir = tempfile.mkdtemp()
        
        # Extract SOP
        print(" Extracting SOP...")
        if sop_file.filename.lower().endswith(".pdf"):
            sop_path = os.path.join(temp_dir, "temp.pdf")
            sop_file.save(sop_path)
            sop_text = extract_text_from_pdf(sop_path)
        else:
            sop_text = sop_file.read().decode('utf-8')
        
        sop_points = extract_sop_points(sop_text)
        print(f" Extracted {len(sop_points)} SOP points")

        # Extract frames (in memory)
        print(" Extracting video frames...")
        frames = extract_frames(video_file)

        # Caption frames
        print(" Analyzing video content...")
        captions_json = caption_frames(frames)

        # Generate analysis report
        print(" Generating analysis report...")
        report = generate_analysis_report(captions_json, sop_points)

        # Generate compliance checklist
        print(" Evaluating SOP compliance...")
        checklist = generate_compliance_checklist(captions_json, sop_points)

        print(" Analysis complete!")

        # Calculate compliance
        satisfied_count = sum(1 for item in checklist if item.get("satisfied", False))
        compliance_rate = (satisfied_count / len(checklist) * 100) if checklist else 0

        return jsonify({
            "status": "success",
            "message": "Analysis complete",
            "sop_points": sop_points,
            "checklist": checklist,
            "report_markdown": report,
            "compliance_rate": round(compliance_rate, 1),
            "satisfied_count": satisfied_count,
            "total_items": len(checklist)
        })

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Manufacturing Video Analysis System")
    print("=" * 50)
    print(f"📱 Server starting on http://0.0.0.0:8000")
    print(f"🖥️ Device: {DEVICE}")
    print(f"🤖 Model: {MODEL_ID}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)