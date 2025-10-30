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
    raise ValueError(" GROQ_API_KEY missing in environment vars.")
client = Groq(api_key=GROQ_API_KEY)

# Flask setup
app = Flask(__name__)

def extract_sop_points(sop_text):
    """Split SOP text into individual checklist points organized by steps."""
    lines = [line.strip() for line in sop_text.split('\n') if line.strip()]
    points = []
    current_step = "General"
    current_point = ""
    step_counter = 1
    
    for line in lines:
        # Check for step headers
        step_match = re.match(r'^(?:step|section|phase)\s*[-:#.]?\s*(\d+|[a-z])/i', line, re.IGNORECASE)
        if step_match or line.strip().lower().startswith(('step', 'section', 'phase')):
            current_step = f"Step {step_counter}: {line.split(':', 1)[-1].strip()}" if ':' in line else line.strip()
            step_counter += 1
            if current_point:  # Save any pending point
                points.append({
                    'step': current_step,
                    'point': current_point.strip()
                })
                current_point = ""
            continue
            
        # Check for bullet points or numbered items
        if line and (line[0].isdigit() or line[0] in '-*•○▪►' or re.match(r'^[a-z][.)]\s', line)):
            if current_point:  # Save previous point if exists
                points.append({
                    'step': current_step,
                    'point': current_point.strip()
                })
            current_point = line.lstrip('0123456789.-*•○▪► abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ).').strip()
        else:
            # Continue current point or start new one
            if current_point:
                current_point += " " + line
            else:
                current_point = line
    
    # Add final point if exists
    if current_point:
        points.append({
            'step': current_step,
            'point': current_point.strip()
        })
    
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

        print(f" Extracting frames (FPS: {fps})")
        
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
    """Generate analysis report using Groq that includes compliance analysis."""
    # Group points by step for better organization
    steps = {}
    for point in sop_points:
        if point['step'] not in steps:
            steps[point['step']] = []
        steps[point['step']].append(point['point'])
    
    # Simplified prompt for more reliable parsing
    prompt = f"""
You are a professional manufacturing process auditor. Analyze the video observations against the SOP requirements.
Return ONLY a valid JSON object matching this exact structure (no additional text or formatting):

{{
    "report": {{
        "executive_summary": "string",
        "compliance_analysis": [
            {{
                "step": "string",
                "requirements": ["string"],
                "observations": ["string"],
                "compliance_status": "compliant|non-compliant|partial",
                "evidence": "string",
                "issues": ["string"],
                "recommendations": ["string"]
            }}
        ],
        "safety_assessment": ["string"],
        "quality_findings": ["string"],
        "recommendations": ["string"],
        "overall_compliance_rate": "number%"
    }}
}}

Video Observations:
{json.dumps(captions_json, indent=2)}

SOP Requirements:
{json.dumps(steps, indent=2)}
"""

    print(" Generating comprehensive analysis...")
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a manufacturing auditor. Return ONLY valid JSON matching the specified structure."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=3000,
        )

        # Clean the response content
        content = response.choices[0].message.content.strip()
        if content.startswith('{') and content.endswith('}'):
            # Attempt to parse the JSON response
            try:
                result = json.loads(content)
                
                # Convert the JSON response to a formatted markdown report
                markdown_report = f"""
# Manufacturing Process Analysis Report

## Executive Summary
{result['report']['executive_summary']}

## Compliance Analysis
"""
                for step in result['report']['compliance_analysis']:
                    markdown_report += f"""
### {step['step']}
- **Requirements:** {', '.join(step['requirements'])}
- **Observations:** {', '.join(step['observations'])}
- **Status:** {step['compliance_status']}
- **Evidence:** {step['evidence']}
- **Issues:** {', '.join(step['issues'])}
- **Recommendations:** {', '.join(step['recommendations'])}
"""

                markdown_report += f"""
## Safety Assessment
{chr(10).join('- ' + item for item in result['report']['safety_assessment'])}

## Quality Control Findings
{chr(10).join('- ' + item for item in result['report']['quality_findings'])}

## Recommendations
{chr(10).join('- ' + item for item in result['report']['recommendations'])}

## Overall Compliance
{result['report']['overall_compliance_rate']}
"""

                return {
                    'markdown': markdown_report,
                    'compliance_data': result['report']['compliance_analysis'],
                    'compliance_rate': float(result['report']['overall_compliance_rate'].rstrip('%'))
                }

            except json.JSONDecodeError:
                print(" Error parsing LLM response")
                return {
                    'markdown': "Error generating report",
                    'compliance_data': [],
                    'compliance_rate': 0
                }
        else:
            print(" Invalid response format")
            return {
                'markdown': "Error: Invalid response format",
                'compliance_data': [],
                'compliance_rate': 0
            }

    except Exception as e:
        print(f"Error generating report: {e}")
        return {
            'markdown': "Error generating report",
            'compliance_data': [],
            'compliance_rate': 0
        }

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

        if not sop_file.filename.lower().endswith('.txt'):
            return jsonify({"error": "Only .txt files are supported for SOP."}), 400

        print(f" Processing files: {video_file.filename}, {sop_file.filename}")

        # Extract SOP text directly from txt file
        print(" Reading SOP...")
        sop_text = sop_file.read().decode('utf-8')
        sop_points = extract_sop_points(sop_text)
        print(f" Extracted {len(sop_points)} SOP points")

        # Extract frames (in memory)
        print(" Extracting video frames...")
        frames = extract_frames(video_file)

        # Caption frames
        print(" Analyzing video content...")
        captions_json = caption_frames(frames)

        # Generate combined analysis report
        print(" Generating analysis report...")
        analysis_result = generate_analysis_report(captions_json, sop_points)

        print(" Analysis complete!")

        return jsonify({
            "status": "success",
            "message": "Analysis complete",
            "sop_points": sop_points,
            "checklist": analysis_result['compliance_data'],
            "report_markdown": analysis_result['markdown'],
            "compliance_rate": analysis_result['compliance_rate'],
            "satisfied_count": len([x for x in analysis_result['compliance_data'] if x['compliance_status'].lower() == 'compliant']),
            "total_items": len(analysis_result['compliance_data'])
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
