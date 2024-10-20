from flask import Flask, request, render_template, redirect, url_for, jsonify
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os
import time
from werkzeug.utils import secure_filename
from cassandra.cluster import Cluster

# Define the Flask app
app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# List of class names corresponding to the model's output indices
class_names = [
    'Aerosol Cans', 'Aluminum Food Cans', 'Aluminum Soda Cans',
    'Cardboard Boxes', 'Cardboard Packaging', 'Clothing', 'Coffee Grounds',
    'Disposable Plastic Cutlery', 'Eggshells', 'Food Waste', 'Glass Beverage Bottles',
    'Glass Cosmetic Container', 'Glass Food Jars', 'Magazines', 'Newspaper',
    'Office Paper', 'Paper Cups', 'Plastic Cup Lids', 'Plastic Detergent Bottles',
    'Plastic Food Containers', 'Plastic Shopping Bags', 'Plastic Soda Bottles',
    'Plastic Straws', 'Plastic Trash Bags', 'Plastic Water Bottles', 'Shoes',
    'Steel Food Cans', 'Styrofoam Cups', 'Styrofoam Food Containers', 'Tea Bags'
]

# Define your model architecture (ResNet example)
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# Load the model
model = ResNet(num_classes=len(class_names))
model_file = 'app/model.pth'
model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])



# Function to predict the class of an image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transformations(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        class_name = class_names[predicted_class]
        return predicted_class, class_name  # Return both index and name

# Function to get material analysis from Cassandra
def get_material_analysis(material):
    analysis = {
        'biodegradability': [],
        'recyclability': [],
        'decomposition_time': []
    }

    # Define and execute the queries using parameterized queries
    biodegradability_query = "SELECT material, biodegradability_rate FROM waste_data WHERE material = %s;"
    recyclability_query = "SELECT material, recyclability_rate FROM waste_data WHERE material = %s;"
    decomposition_time_query = "SELECT material, decomposition_time FROM waste_data WHERE material = %s;"

    biodegradability_rows = session.execute(biodegradability_query, [material])
    recyclability_rows = session.execute(recyclability_query, [material])
    decomposition_time_rows = session.execute(decomposition_time_query, [material])

    for row in biodegradability_rows:
        analysis['biodegradability'].append({
            'material': row.material,
            'biodegradability_rate': row.biodegradability_rate
        })

    for row in recyclability_rows:
        analysis['recyclability'].append({
            'material': row.material,
            'recyclability_rate': row.recyclability_rate
        })

    for row in decomposition_time_rows:
        analysis['decomposition_time'].append({
            'material': row.material,
            'decomposition_time': row.decomposition_time
        })

    return analysis

# Define the queries for biodegradability based on predicted material
def get_biodegradability_queries(predicted_material):
    low_biodegradability_query = f"SELECT material, biodegradability_rate FROM waste_data WHERE biodegradability_rate <= 30 AND material = '{predicted_material}' ALLOW FILTERING;"
    moderate_biodegradability_query = f"SELECT material, biodegradability_rate FROM waste_data WHERE biodegradability_rate > 30 AND biodegradability_rate <= 70 AND material = '{predicted_material}' ALLOW FILTERING;"
    high_biodegradability_query = f"SELECT material, biodegradability_rate FROM waste_data WHERE biodegradability_rate > 70 AND material = '{predicted_material}' ALLOW FILTERING;"
    return low_biodegradability_query, moderate_biodegradability_query, high_biodegradability_query

# Define the queries for recyclability based on predicted material
def get_recyclability_queries(predicted_material):
    low_recyclability_query = f"SELECT material, recyclability_rate FROM waste_data WHERE recyclability_rate <= 30 AND material = '{predicted_material}' ALLOW FILTERING;"
    moderate_recyclability_query = f"SELECT material, recyclability_rate FROM waste_data WHERE recyclability_rate > 30 AND recyclability_rate <= 70 AND material = '{predicted_material}' ALLOW FILTERING;"
    high_recyclability_query = f"SELECT material, recyclability_rate FROM waste_data WHERE recyclability_rate > 70 AND material = '{predicted_material}' ALLOW FILTERING;"
    return low_recyclability_query, moderate_recyclability_query, high_recyclability_query

# Define the queries for decomposition time based on predicted material
def get_decomposition_time_queries(predicted_material):
    low_decomposition_time_query = f"SELECT material, decomposition_time FROM waste_data WHERE decomposition_time > 0 AND decomposition_time < 90 AND material = '{predicted_material}' ALLOW FILTERING;"
    moderate_decomposition_time_query = f"SELECT material, decomposition_time FROM waste_data WHERE decomposition_time > 90 AND decomposition_time < 365000 AND material = '{predicted_material}' ALLOW FILTERING;"
    high_decomposition_time_query = f"SELECT material, decomposition_time FROM waste_data WHERE decomposition_time > 365000 AND material = '{predicted_material}' ALLOW FILTERING;"
    return low_decomposition_time_query, moderate_decomposition_time_query, high_decomposition_time_query

# Function to fetch categorized results based on the predicted class (material)
def fetch_material_data(predicted_class_name):
    # Fetch the queries based on the predicted class name
    low_biodegradability_query, moderate_biodegradability_query, high_biodegradability_query = get_biodegradability_queries(predicted_class_name)
    low_recyclability_query, moderate_recyclability_query, high_recyclability_query = get_recyclability_queries(predicted_class_name)
    low_decomposition_time_query, moderate_decomposition_time_query, high_decomposition_time_query = get_decomposition_time_queries(predicted_class_name)

    # Execute the queries
    low_biodegradability_rows = session.execute(low_biodegradability_query)
    moderate_biodegradability_rows = session.execute(moderate_biodegradability_query)
    high_biodegradability_rows = session.execute(high_biodegradability_query)

    low_recyclability_rows = session.execute(low_recyclability_query)
    moderate_recyclability_rows = session.execute(moderate_recyclability_query)
    high_recyclability_rows = session.execute(high_recyclability_query)

    low_decomposition_time_rows = session.execute(low_decomposition_time_query)
    moderate_decomposition_time_rows = session.execute(moderate_decomposition_time_query)
    high_decomposition_time_rows = session.execute(high_decomposition_time_query)

    # Collect results and categorize
    biodegradability_materials = []
    recyclability_materials = []
    decomposition_materials = []

    # Biodegradability Categories
    for row in low_biodegradability_rows:
        biodegradability_materials.append({
            'material': row.material,
            'biodegradability_rate': row.biodegradability_rate,
            'category': 'Low Biodegradability'
        })

    for row in moderate_biodegradability_rows:
        biodegradability_materials.append({
            'material': row.material,
            'biodegradability_rate': row.biodegradability_rate,
            'category': 'Moderate Biodegradability'
        })

    for row in high_biodegradability_rows:
        biodegradability_materials.append({
            'material': row.material,
            'biodegradability_rate': row.biodegradability_rate,
            'category': 'High Biodegradability'
        })

    # Recyclability Categories
    for row in low_recyclability_rows:
        recyclability_materials.append({
            'material': row.material,
            'recyclability_rate': row.recyclability_rate,
            'category': 'Low Recyclability'
        })

    for row in moderate_recyclability_rows:
        recyclability_materials.append({
            'material': row.material,
            'recyclability_rate': row.recyclability_rate,
            'category': 'Moderate Recyclability'
        })

    for row in high_recyclability_rows:
        recyclability_materials.append({
            'material': row.material,
            'decomposition_time': row.decomposition_time,
            'category': 'Low Decomposition Time'
           })
    
    for row in low_decomposition_time_rows:
        decomposition_materials.append({
        'material': row.material,
        'decomposition_time': row.decomposition_time,
        'category': 'Low Decomposition Time'
    })
    
    for row in moderate_decomposition_time_rows:
        decomposition_materials.append({
        'material': row.material,
        'decomposition_time': row.decomposition_time,
        'category': 'Moderate Decomposition Time'
    })

    for row in high_decomposition_time_rows:
         decomposition_materials.append({
        'material': row.material,
        'decomposition_time': row.decomposition_time,
        'category': 'High Decomposition Time'
    })

    return {
      'biodegradability': biodegradability_materials,
      'recyclability': recyclability_materials,
      'decomposition_time': decomposition_materials
    }

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict the class of the uploaded image
        predicted_class, predicted_class_name = predict_image(filepath)

        # Fetch material data based on the predicted class name
        material_data = fetch_material_data(predicted_class_name)

        # Render the results page with the prediction and material data
        return render_template('results.html', 
                               filename=filename, 
                               predicted_class_name=predicted_class_name,
                               material_data=material_data)

    return redirect(request.url)

# Route to display uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Function to check if the file type is allowed
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Run the app
if __name__ == "__main__":
    app.run(debug=True)