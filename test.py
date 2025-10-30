import os
import cv2
import json
import torch
import tempfile
import shutil
import re
from PIL import Image
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template
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
    """Generate structured compliance checklist using Groq - passes checkpoints directly to LLM."""
    
    print(f"🔍 Analyzing {len(sop_points)} SOP points...")
    
    # Create a detailed list of all SOP points with their steps
    sop_requirements = []
    for point in sop_points:
        sop_requirements.append({
            "step": point['step'],
            "point": point['point']
        })
    
    # Single comprehensive prompt with all checkpoints
    prompt = f"""
You are a strict manufacturing compliance auditor. Your task is to evaluate video observations against SOP requirements.

VIDEO OBSERVATIONS (with timestamps):
{json.dumps(captions_json, indent=2)}

SOP REQUIREMENTS:
{json.dumps(sop_requirements, indent=2)}

TASK: For EACH requirement in the SOP, determine if it was satisfied based on the video observations.

You MUST wrap your response in a JSON object with a "checklist" key containing the array:
{{
    "checklist": [
        {{
            "step": "step name from SOP",
            "point": "exact requirement text from SOP",
            "satisfied": true or false,
            "evidence": "specific detailed observation from video describing what was seen",
            "timestamps": ["00:00", "00:05"]
        }}
    ]
}}

RULES:
1. Return ONLY the JSON array - no markdown, no explanation, no text before or after
2. Include ALL requirements from the SOP (all {len(sop_requirements)} items)
3. Be strict: only mark satisfied=true if there is CLEAR, SPECIFIC evidence in the observations
4. If no evidence found, set satisfied=false and evidence="No evidence observed in video"
5. Always include timestamps from the observations when evidence exists
6. For evidence field: provide a detailed description of what was observed (2-3 sentences minimum when satisfied=true)
7. Copy the exact "step" and "point" text from the SOP requirements provided
"""

    try:
        print("🤖 Sending all checkpoints to LLM for evaluation...")
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a strict compliance auditor. Return ONLY valid JSON array, no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )
        
        result_raw = response.choices[0].message.content.strip()
        print("📄 Raw response received (first 500 chars):", result_raw[:500])
        
        # Clean markdown formatting if present
        if result_raw.startswith("```"):
            result_raw = result_raw.split("```", 1)[1]
            if result_raw.startswith("json"):
                result_raw = result_raw[4:]
            if "```" in result_raw:
                result_raw = result_raw.split("```")[0]
            result_raw = result_raw.strip()
        
        # Parse JSON - handle both array and object formats
        parsed_data = json.loads(result_raw)
        
        # If it's an object with a key containing the array, extract it
        if isinstance(parsed_data, dict):
            # Look for common keys that might contain the array
            for key in ['checklist', 'items', 'results', 'data', 'compliance']:
                if key in parsed_data and isinstance(parsed_data[key], list):
                    checklist = parsed_data[key]
                    break
            else:
                # If no known key found, try to find any list value
                for value in parsed_data.values():
                    if isinstance(value, list):
                        checklist = value
                        break
                else:
                    raise ValueError("Could not find array in response object")
        else:
            checklist = parsed_data
        
        print(f"✅ Successfully parsed {len(checklist)} compliance items")
        
        # Validate each item has required fields
        validated_checklist = []
        for item in checklist:
            validated_item = {
                "step": item.get("step", "Unknown"),
                "point": item.get("point", "Unknown requirement"),
                "satisfied": item.get("satisfied", False),
                "evidence": item.get("evidence", "No evidence provided"),
                "timestamps": item.get("timestamps", [])
            }
            validated_checklist.append(validated_item)
        
        return validated_checklist
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Raw response: {result_raw[:1000] if 'result_raw' in locals() else 'No response'}")
        # Return error items for all SOP points
        return [{
            "step": point['step'],
            "point": point['point'],
            "satisfied": False,
            "evidence": "Error: Could not parse LLM response",
            "timestamps": []
        } for point in sop_points]
        
    except Exception as e:
        print(f"❌ Error during compliance check: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return error items for all SOP points
        return [{
            "step": point['step'],
            "point": point['point'],
            "satisfied": False,
            "evidence": f"Error: {str(e)}",
            "timestamps": []
        } for point in sop_points]

@app.route("/")
def index():
    """Serve main page."""
    return render_template("index2.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint."""
    try:
        # Get uploaded files
        video_file = request.files.get("video")
        sop_file = request.files.get("sop")

        if not video_file or not sop_file:
            return jsonify({"error": "Both video and SOP files are required."}), 400

        # Validate SOP file type - only TXT allowed
        if not sop_file.filename.lower().endswith(".txt"):
            return jsonify({"error": "Only TXT files are allowed for SOP. Please upload a .txt file."}), 400

        print(f" Processing files: {video_file.filename}, {sop_file.filename}")

        # Extract SOP from TXT file
        print(" Reading SOP from TXT file...")
        sop_text = sop_file.read().decode('utf-8')
        
        if not sop_text.strip():
            return jsonify({"error": "SOP file is empty. Please provide a valid TXT file with SOP content."}), 400
        
        sop_points = extract_sop_points(sop_text)
        
        if not sop_points:
            return jsonify({"error": "Could not extract any SOP points from the file. Please check the file format."}), 400
            
        print(f" Extracted {len(sop_points)} SOP points")

        # Extract frames (in memory)
        print(" Extracting video frames...")
        frames = extract_frames(video_file)

        # Caption frames
        print(" Analyzing video content...")
        captions_json = caption_frames(frames)

        # Generate compliance checklist - LLM does the comparison
        print(" Evaluating SOP compliance with LLM...")
        checklist = generate_compliance_checklist(captions_json, sop_points)

        # Generate analysis report
        print(" Generating analysis report...")
        report = generate_analysis_report(captions_json, sop_points)

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
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Manufacturing Video Analysis System")
    print("=" * 50)
    print(f"📱 Server starting on http://0.0.0.0:8000")
    print(f"🖥️ Device: {DEVICE}")
    print(f"🤖 Model: {MODEL_ID}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
