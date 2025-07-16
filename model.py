import torch
from torchvision import transforms
from PIL import Image
import io

# ==== Device Configuration ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Crop Or Not Crop Detection Model ====
crop_or_not_model = torch.load("app/models/crop_or_notcrop_model.pth", map_location=device, weights_only=False)
crop_or_not_model.eval()

# ==== Load Crop Detection Model ====
crop_detection_model = torch.load("app/models/crop_detection_newmodel2.pth", map_location=device, weights_only=False)
crop_detection_model.eval()

# ==== Load Disease Models ====
model_paths = {
    "Banana": "app/models/Banana_model.pth",
    "Cotton": "app/models/Cotton_model.pth",
    "Maize": "app/models/Maize_model.pth",
    "Mango": "app/models/mango_newmodel2.pth",
    "Onion": "app/models/Onion_model.pth",
    "Paddy": "app/models/Paddy_model.pth",
    "Potato": "app/models/Potato_model.pth",
    "Sugarcane": "app/models/Sugarcane_model.pth",
    "Tomato": "app/models/Tomato_model.pth",
    "Wheat": "app/models/Wheat_model.pth",
}

sub_models = {}
for crop, path in model_paths.items():
    try:
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        sub_models[crop] = model
    except Exception as e:
        print(f"⚠️ Could not load model for {crop}: {e}")
        sub_models[crop] = None

# ==== Define Transform ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Class Names ====
crop_or_not_classes = ['Crop', 'Not Crop']
class_names = ['Banana', 'Cotton', 'Maize', 'Mango', 'Not Crop', 'Onion', 'Paddy', 'Potato', 'Sugarcane', 'Tomato', 'Wheat']
banana_classes = ['Anthracnose', 'Scarring Beetle', 'Banana Skipper Damage', 'Banana Split Peel',
                  'Black and Yellow Sigatoka', 'Chewing insect damage on banana leaf',
                  'Healthy Banana', 'Healthy Banana leaf', 'Panama Wilt Disease']
cotton_classes = ['Aphids', 'Army worm', 'Bacterial Blight', 'Healthy', 'Powdery Mildew', 'Target spot']
maize_classes = ['Common Rust', 'Gray Leaf Spot', 'Healthy', 'Northern Leaf Blight']
mango_classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Gall Midge',
                 'Healthy', 'Powdery Mildew', 'Sooty Mould', 'die back']
onion_classes = ['Iris yellow virus', 'Stemphylium leaf blight and collectrichum leaf blight',
                 'healthy', 'purple blotch']
paddy_classes = ['Leafsmut', 'bacterial_leaf_blight ', 'blast', 'brown_spot', 'dead_heart',
                 'downy_mildew', 'hispa', 'healthy', 'tungro']
potato_classes = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']
sugarcane_classes = ['BacterialBlights', 'Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
tomato_classes = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Late_Mold', 'Septoria_leaf_spot',
                  'Spider_mites Two-spotted_spider_mite', 'Target_Spot',
                  'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy', 'powdery_mildew']
wheat_classes = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot',
                 'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew',
                 'Mite', 'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust']

disease_classes = {
    "Banana": banana_classes,
    "Cotton": cotton_classes,
    "Maize": maize_classes,
    "Mango": mango_classes,
    "Onion": onion_classes,
    "Paddy": paddy_classes,
    "Potato": potato_classes,
    "Sugarcane": sugarcane_classes,
    "Tomato": tomato_classes,
    "Wheat": wheat_classes,
}


# ==== Prediction Function ====
def predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {'result': False, 'message': 'Invalid image format.'}
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict the whether the image is a crop or not
    with torch.no_grad():
        crop_or_not_output = crop_or_not_model(input_tensor)
        predicted_index = torch.argmax(crop_or_not_output, 1).item()
        result = crop_or_not_classes[predicted_index]
    if result == 'Not Crop':
        msg = {'result': False, 'message': 'The image does not contain a crop.'}
        return msg
    else:
        with torch.no_grad():
            crop_output = crop_detection_model(input_tensor)
            predicted_crop_index = torch.argmax(crop_output, 1).item()
            predicted_crop = class_names[predicted_crop_index]

            # If crop has subclassification model
            if predicted_crop in sub_models and sub_models[predicted_crop] is not None:
                disease_output = sub_models[predicted_crop](input_tensor)
                predicted_index = torch.argmax(disease_output, 1).item()
                predicted_class = disease_classes[predicted_crop][predicted_index]
            else:
                predicted_class = predicted_crop  # fallback
            msg = {'result': True, 'message': 'Prediction successful.', 
                   'crop': predicted_crop, 'disease': predicted_class}
            return msg
