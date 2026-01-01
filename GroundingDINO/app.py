import streamlit as st
import os
import shutil
import torch
from groundingdino.util.inference import load_model, load_image, predict
import zipfile
import tempfile
from PIL import Image, ImageDraw
import sys

# Add pipeline script to path
PIPELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "Dataset-generation-and-annotation-framework-using-open-vocab-vision-language-model", 
                           "scripts")
sys.path.append(PIPELINE_DIR)

try:
    from pipeline import run_pipeline
except ImportError:
    pass # Will handle error if used

# Set page config
st.set_page_config(page_title="Auto Labeler", layout="wide")

# Constants
# Assuming app.py is in ann_el/GroundingDINO/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "groundingdino_swint_ogc.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

@st.cache_resource
def load_dino_model():
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"Weights not found at {WEIGHTS_PATH}. Please download them.")
        return None
    if not os.path.exists(CONFIG_PATH):
        st.error(f"Config not found at {CONFIG_PATH}.")
        return None
    
    try:
        model = load_model(CONFIG_PATH, WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_images(source_type, source_data, class_labels, box_threshold, text_threshold, model):
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input_images")
        output_dir = os.path.join(temp_dir, "dataset")
        images_out = os.path.join(output_dir, "images")
        labels_out = os.path.join(output_dir, "labels")
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)
        
        # Prepare images based on source
        image_paths = []
        if source_type == "upload":
            for uploaded_file in source_data:
                file_path = os.path.join(input_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_paths.append(file_path)
        elif source_type == "scrape":
            # source_data is the directory path containing scraped images
            if os.path.exists(source_data):
                for f in os.listdir(source_data):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        src = os.path.join(source_data, f)
                        dst = os.path.join(input_dir, f)
                        shutil.copy2(src, dst)
                        image_paths.append(dst)
            
        # Create data.yaml
        yaml_content = f"""train: ../images
val: ../images
nc: {len(class_labels)}
names: {class_labels}"""
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
            
        # Construct prompt
        text_prompt = " . ".join(class_labels) + " ."
        
        # Process
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            status_text.text(f"Processing {img_name}...")
            
            try:
                image_source, image = load_image(img_path)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                
                label_lines = []
                for box, phrase in zip(boxes, phrases):
                    class_id = -1
                    for idx, label in enumerate(class_labels):
                        if label.lower() in phrase.lower() or phrase.lower() in label.lower():
                            class_id = idx
                            break
                    
                    if class_id != -1:
                        # YOLO format: class_id center_x center_y width height
                        line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                        label_lines.append(line)
                
                # Save if detections found
                if label_lines:
                    label_file = os.path.splitext(img_name)[0] + ".txt"
                    with open(os.path.join(labels_out, label_file), 'w') as f:
                        f.write("\n".join(label_lines))
                    
                    # Copy image
                    shutil.copy(img_path, os.path.join(images_out, img_name))
                    
            except Exception as e:
                st.warning(f"Failed to process {img_name}: {e}")
            
            progress_bar.progress((i + 1) / len(image_paths))
            
        status_text.text("Processing complete! Generating preview...")
        
        # Show preview of annotated images
        st.subheader("Annotated Images Preview")
        preview_cols = st.columns(4)
        processed_files = sorted(os.listdir(images_out))[:4]
        
        for idx, img_file in enumerate(processed_files):
            img_path = os.path.join(images_out, img_file)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_out, label_file)
            
            if os.path.exists(label_path):
                try:
                    img = Image.open(img_path)
                    draw = ImageDraw.Draw(img)
                    w, h = img.size
                    
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:])
                            
                            # Convert YOLO to xyxy
                            x1 = (cx - bw/2) * w
                            y1 = (cy - bh/2) * h
                            x2 = (cx + bw/2) * w
                            y2 = (cy + bh/2) * h
                            
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            # Optional: Add label text
                            if 0 <= cls_id < len(class_labels):
                                draw.text((x1, y1), class_labels[cls_id], fill="red")
                    
                    with preview_cols[idx % 4]:
                        st.image(img, caption=img_file, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not preview {img_file}: {e}")

        status_text.text("Zipping files...")
        
        # Zip the dataset directory
        zip_path = os.path.join(temp_dir, "yolo_dataset.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
                    
        # Read zip file to bytes
        with open(zip_path, "rb") as f:
            zip_data = f.read()
            
        return zip_data

# Main UI
st.title("📷 Auto-Labeler for YOLO")
st.markdown("Upload images or scrape from web, define classes, and get a labeled YOLO dataset using GroundingDINO.")

with st.sidebar:
    st.header("Configuration")
    box_threshold = st.slider("Box Threshold", 0.1, 0.9, 0.35, 0.05)
    text_threshold = st.slider("Text Threshold", 0.1, 0.9, 0.25, 0.05)
    
    st.info("Ensure you have the model weights downloaded in the `weights` folder.")

# Data Source Selection
data_source = st.radio("Data Source", ["Upload Images", "Web Scrape"], horizontal=True)

source_data = None
uploaded_files = None
scrape_query = None
scrape_num = 50

if data_source == "Upload Images":
    uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        scrape_query = st.text_input("Search Query", placeholder="e.g., person walking dog")
    with col2:
        scrape_num = st.number_input("Number of Images", min_value=1, max_value=500, value=10)
    
    st.info("Web scraping uses Bing (free) or APIs if configured in environment variables.")

class_input = st.text_input("Enter Class Labels (comma separated)", placeholder="eyes, nose, mouth")

if st.button("Generate Dataset"):
    valid_input = False
    source_type = ""
    
    if data_source == "Upload Images":
        if not uploaded_files:
            st.error("Please upload some images.")
        else:
            valid_input = True
            source_data = uploaded_files
            source_type = "upload"
    else:
        if not scrape_query:
            st.error("Please enter a search query.")
        else:
            valid_input = True
            source_type = "scrape"
            
    if valid_input:
        if not class_input:
            st.error("Please enter class labels.")
        else:
            class_labels = [label.strip() for label in class_input.split(",") if label.strip()]
            
            # Run scraping if needed
            if source_type == "scrape":
                with st.spinner(f"Scraping images for '{scrape_query}'..."):
                    try:
                        # Load config from env
                        config = {}
                        if os.getenv('SERPAPI_KEY'): config['serpapi_key'] = os.getenv('SERPAPI_KEY')
                        if os.getenv('FLICKR_KEY'): config['flickr_key'] = os.getenv('FLICKR_KEY')
                        if os.getenv('UNSPLASH_KEY'): config['unsplash_key'] = os.getenv('UNSPLASH_KEY')
                        if os.getenv('USE_SELENIUM'): config['use_selenium'] = True
                        
                        # Run pipeline
                        # run_pipeline returns the output directory path
                        if 'run_pipeline' in globals():
                            source_data = str(run_pipeline(
                                query=scrape_query,
                                num=scrape_num,
                                min_size=200,
                                task='auto',
                                config=config
                            ))
                            st.success(f"Scraped images saved to {source_data}")
                            
                            # Show preview of scraped images
                            st.subheader("Scraped Images Preview")
                            if os.path.exists(source_data):
                                scraped_images = [f for f in os.listdir(source_data) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                                preview_cols = st.columns(4)
                                for idx, img_file in enumerate(scraped_images[:4]):
                                    img_path = os.path.join(source_data, img_file)
                                    with preview_cols[idx % 4]:
                                        st.image(img_path, caption=img_file, use_container_width=True)
                        else:
                            st.error("Pipeline module not loaded. Cannot scrape.")
                            valid_input = False
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")
                        valid_input = False

            if valid_input:
                model = load_dino_model()
                if model:
                    with st.spinner("Running GroundingDINO..."):
                        zip_data = process_images(source_type, source_data, class_labels, box_threshold, text_threshold, model)
                        
                    if zip_data:
                        st.success("Dataset generated successfully!")
                        st.download_button(
                            label="Download YOLO Dataset (ZIP)",
                            data=zip_data,
                            file_name="yolo_dataset.zip",
                            mime="application/zip"
                        )
