"""
OptiCrop - Smart Agricultural Production Optimization Engine
Flask Web Application
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'opticrop_secret_key_2024'

# ============================================================
# LOAD MODELS
# ============================================================
MODEL_PATH = 'crop_recommendation_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'
DATASET_PATH = 'Crop_recommendation.csv'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Models loaded successfully")
except FileNotFoundError:
    print("✗ Model files not found. Run model_training.py first.")
    model = None
    scaler = None
    label_encoder = None

# Load dataset for analysis
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✓ Dataset loaded: {df.shape}")
except FileNotFoundError:
    print("✗ Dataset not found.")
    df = None

# ============================================================
# CROP INFORMATION DATABASE
# ============================================================
CROP_INFO = {
    'rice': {
        'season': 'Kharif (Rainy)',
        'water_need': 'High (1200-1400 mm)',
        'soil_type': 'Clay or loamy',
        'temperature_range': '20-35°C',
        'growing_period': '90-120 days',
        'description': 'Rice is a staple food crop grown in warm, humid conditions with high water availability. It thrives in clay soils that retain water.'
    },
    'maize': {
        'season': 'Kharif & Rabi',
        'water_need': 'Moderate (500-800 mm)',
        'soil_type': 'Well-drained loamy',
        'temperature_range': '18-30°C',
        'growing_period': '90-110 days',
        'description': 'Maize (corn) is a versatile crop used for food, feed, and industrial purposes. It requires well-drained soil and moderate rainfall.'
    },
    'chickpea': {
        'season': 'Rabi (Winter)',
        'water_need': 'Low (300-400 mm)',
        'soil_type': 'Sandy loam to clay',
        'temperature_range': '10-25°C',
        'growing_period': '90-120 days',
        'description': 'Chickpea is a protein-rich legume grown in cool, dry conditions. It enriches soil nitrogen through biological fixation.'
    },
    'kidneybeans': {
        'season': 'Kharif & Rabi',
        'water_need': 'Moderate (400-500 mm)',
        'soil_type': 'Loamy, well-drained',
        'temperature_range': '15-25°C',
        'growing_period': '80-120 days',
        'description': 'Kidney beans are protein-rich legumes requiring moderate temperatures and well-drained soil for optimal growth.'
    },
    'pigeonpeas': {
        'season': 'Kharif',
        'water_need': 'Low-Moderate (400-600 mm)',
        'soil_type': 'Sandy loam to clay',
        'temperature_range': '15-35°C',
        'growing_period': '120-180 days',
        'description': 'Pigeonpea (tur/arhar) is a drought-tolerant legume that improves soil fertility through nitrogen fixation.'
    },
    'mothbeans': {
        'season': 'Kharif',
        'water_need': 'Low (200-300 mm)',
        'soil_type': 'Sandy to loamy',
        'temperature_range': '25-35°C',
        'growing_period': '70-90 days',
        'description': 'Mothbean is a drought-resistant legume well-suited for arid and semi-arid regions with minimal rainfall.'
    },
    'mungbean': {
        'season': 'Kharif & Summer',
        'water_need': 'Low-Moderate (300-400 mm)',
        'soil_type': 'Sandy loam to clay',
        'temperature_range': '25-35°C',
        'growing_period': '60-90 days',
        'description': 'Mungbean (green gram) is a short-duration legume rich in protein, commonly grown in tropical regions.'
    },
    'blackgram': {
        'season': 'Kharif & Rabi',
        'water_need': 'Moderate (300-500 mm)',
        'soil_type': 'Sandy loam to clay',
        'temperature_range': '25-35°C',
        'growing_period': '70-90 days',
        'description': 'Blackgram (urad dal) is a protein-rich pulse crop that grows well in diverse soil conditions.'
    },
    'lentil': {
        'season': 'Rabi',
        'water_need': 'Low (250-350 mm)',
        'soil_type': 'Sandy loam to clay',
        'temperature_range': '10-25°C',
        'growing_period': '80-110 days',
        'description': 'Lentil (masoor dal) is a cool-season legume known for its high protein content and nitrogen-fixing ability.'
    },
    'pomegranate': {
        'season': 'Perennial',
        'water_need': 'Moderate (500-700 mm)',
        'soil_type': 'Sandy loam, well-drained',
        'temperature_range': '25-35°C',
        'growing_period': 'Perennial (5-7 months to fruit)',
        'description': 'Pomegranate is a drought-tolerant fruit crop that thrives in semi-arid regions with well-drained soil.'
    },
    'banana': {
        'season': 'Perennial',
        'water_need': 'High (1000-1500 mm)',
        'soil_type': 'Rich loamy, well-drained',
        'temperature_range': '20-35°C',
        'growing_period': '10-12 months',
        'description': 'Banana is a tropical fruit crop requiring consistent moisture, rich soil, and warm temperatures throughout the year.'
    },
    'mango': {
        'season': 'Perennial',
        'water_need': 'Moderate (400-600 mm)',
        'soil_type': 'Well-drained loamy',
        'temperature_range': '24-30°C',
        'growing_period': 'Perennial (3-6 years to fruit)',
        'description': 'Mango is the king of fruits, thriving in tropical climates with a distinct dry season for flowering.'
    },
    'grapes': {
        'season': 'Perennial',
        'water_need': 'Moderate (500-700 mm)',
        'soil_type': 'Sandy loam, well-drained',
        'temperature_range': '15-35°C',
        'growing_period': 'Perennial (6-8 months to fruit)',
        'description': 'Grapes require well-drained soil, warm days, and cool nights for optimal sugar development in fruits.'
    },
    'watermelon': {
        'season': 'Summer',
        'water_need': 'Moderate (400-600 mm)',
        'soil_type': 'Sandy loam',
        'temperature_range': '20-35°C',
        'growing_period': '70-90 days',
        'description': 'Watermelon is a warm-season fruit crop that requires sandy soil and consistent moisture for sweet fruit development.'
    },
    'muskmelon': {
        'season': 'Summer',
        'water_need': 'Moderate (400-500 mm)',
        'soil_type': 'Sandy loam',
        'temperature_range': '20-35°C',
        'growing_period': '70-90 days',
        'description': 'Muskmelon (cantaloupe) is a warm-season fruit crop grown in sandy, well-drained soils with warm temperatures.'
    },
    'apple': {
        'season': 'Perennial (Temperate)',
        'water_need': 'Moderate (400-600 mm)',
        'soil_type': 'Well-drained loamy',
        'temperature_range': '10-25°C',
        'growing_period': 'Perennial (3-5 years to fruit)',
        'description': 'Apple is a temperate fruit crop requiring cold winters (chilling hours) and moderate summers for quality fruit.'
    },
    'orange': {
        'season': 'Perennial',
        'water_need': 'Moderate (500-800 mm)',
        'soil_type': 'Well-drained loamy',
        'temperature_range': '20-35°C',
        'growing_period': 'Perennial (2-3 years to fruit)',
        'description': 'Orange is a citrus fruit crop that thrives in subtropical climates with well-drained, slightly acidic soil.'
    },
    'papaya': {
        'season': 'Perennial',
        'water_need': 'Moderate (400-600 mm)',
        'soil_type': 'Rich loamy, well-drained',
        'temperature_range': '22-32°C',
        'growing_period': '6-10 months to fruit',
        'description': 'Papaya is a fast-growing tropical fruit crop that requires warm temperatures and consistent moisture.'
    },
    'coconut': {
        'season': 'Perennial',
        'water_need': 'Moderate-High (1000-1500 mm)',
        'soil_type': 'Sandy loam, coastal',
        'temperature_range': '25-35°C',
        'growing_period': 'Perennial (5-7 years to fruit)',
        'description': 'Coconut is a tropical palm crop grown in coastal regions requiring high humidity and consistent rainfall.'
    },
    'cotton': {
        'season': 'Kharif',
        'water_need': 'Moderate (600-800 mm)',
        'soil_type': 'Black cotton soil, loamy',
        'temperature_range': '20-35°C',
        'growing_period': '140-170 days',
        'description': 'Cotton is a cash crop grown for its fiber, requiring a long frost-free period and well-drained black soil.'
    },
    'jute': {
        'season': 'Kharif',
        'water_need': 'High (1000-1200 mm)',
        'soil_type': 'Alluvial, loamy',
        'temperature_range': '20-35°C',
        'growing_period': '90-120 days',
        'description': 'Jute is a natural fiber crop grown in hot, humid conditions with alluvial soil and high rainfall.'
    },
    'coffee': {
        'season': 'Perennial',
        'water_need': 'Moderate (800-1200 mm)',
        'soil_type': 'Well-drained, slightly acidic',
        'temperature_range': '15-28°C',
        'growing_period': 'Perennial (3-4 years to fruit)',
        'description': 'Coffee is a plantation crop grown in shade under specific altitude and temperature conditions.'
    }
}

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    """Crop recommendation page"""
    return render_template('recommend.html')

@app.route('/suitability')
def suitability():
    """Crop suitability assessment page"""
    crop_list = sorted(label_encoder.classes_) if label_encoder is not None else []
    return render_template('suitability.html', crops=crop_list)

@app.route('/research')
def research():
    """Research and analytics page"""
    return render_template('research.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Scenario 1: Smart Crop Recommendation
    Predict the best crop based on soil and environmental parameters.
    """
    if request.method == 'POST':
        try:
            # Get form data
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorous'])
            k = float(request.form['potassium'])
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Validate inputs
            if not (0 <= n <= 200):
                return render_template('result.html', 
                    error='Nitrogen value should be between 0 and 200')
            if not (0 <= p <= 200):
                return render_template('result.html', 
                    error='Phosphorous value should be between 0 and 200')
            if not (0 <= k <= 200):
                return render_template('result.html', 
                    error='Potassium value should be between 0 and 200')
            if not (0 <= temp <= 50):
                return render_template('result.html', 
                    error='Temperature should be between 0 and 50°C')
            if not (0 <= humidity <= 100):
                return render_template('result.html', 
                    error='Humidity should be between 0 and 100%')
            if not (0 <= ph <= 14):
                return render_template('result.html', 
                    error='pH should be between 0 and 14')
            if not (0 <= rainfall <= 500):
                return render_template('result.html', 
                    error='Rainfall should be between 0 and 500 mm')
            
            # Prepare input array
            input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction_encoded = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Decode prediction
            predicted_crop = label_encoder.inverse_transform(prediction_encoded)[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
            top_3_crops = [(label_encoder.inverse_transform([idx])[0], 
                           prediction_proba[0][idx] * 100) for idx in top_3_indices]
            
            # Get crop info
            crop_info = CROP_INFO.get(predicted_crop, {})
            confidence = round(prediction_proba[0][prediction_encoded[0]] * 100, 2)
            
            # Get ideal conditions from dataset
            if df is not None:
                crop_data = df[df['label'] == predicted_crop]
                ideal_conditions = {
                    'N': f"{crop_data['N'].mean():.1f}",
                    'P': f"{crop_data['P'].mean():.1f}",
                    'K': f"{crop_data['K'].mean():.1f}",
                    'temperature': f"{crop_data['temperature'].mean():.1f}",
                    'humidity': f"{crop_data['humidity'].mean():.1f}",
                    'ph': f"{crop_data['ph'].mean():.1f}",
                    'rainfall': f"{crop_data['rainfall'].mean():.1f}"
                }
            else:
                ideal_conditions = None
            
            return render_template('result.html',
                prediction=predicted_crop,
                confidence=confidence,
                top_3=top_3_crops,
                crop_info=crop_info,
                ideal_conditions=ideal_conditions,
                user_input={
                    'N': n, 'P': p, 'K': k,
                    'temperature': temp, 'humidity': humidity,
                    'ph': ph, 'rainfall': rainfall
                },
                scenario='recommend')
                
        except Exception as e:
            return render_template('result.html', 
                error=f'Prediction error: {str(e)}')

@app.route('/check_suitability', methods=['POST'])
def check_suitability():
    """
    Scenario 2: Crop Suitability and Environmental Assessment
    Check if conditions are suitable for a specific crop.
    """
    if request.method == 'POST':
        try:
            crop = request.form['crop']
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorous'])
            k = float(request.form['potassium'])
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Get ideal conditions from dataset
            if df is not None and crop in df['label'].values:
                crop_data = df[df['label'] == crop]
                ideal = {
                    'N': crop_data['N'].mean(),
                    'P': crop_data['P'].mean(),
                    'K': crop_data['K'].mean(),
                    'temperature': crop_data['temperature'].mean(),
                    'humidity': crop_data['humidity'].mean(),
                    'ph': crop_data['ph'].mean(),
                    'rainfall': crop_data['rainfall'].mean()
                }
                
                # Calculate deviation percentages
                deviations = {}
                overall_score = 0
                num_params = 7
                
                for param, user_val in zip(
                    ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
                    [n, p, k, temp, humidity, ph, rainfall]
                ):
                    ideal_val = ideal[param]
                    if ideal_val != 0:
                        deviation = abs(user_val - ideal_val) / ideal_val * 100
                    else:
                        deviation = abs(user_val - ideal_val) * 10 if user_val != 0 else 0
                    
                    # Score: 0% deviation = 100, 50%+ deviation = 0
                    param_score = max(0, 100 - deviation * 2)
                    deviations[param] = {
                        'user': user_val,
                        'ideal': round(ideal_val, 2),
                        'deviation': round(deviation, 1),
                        'score': round(param_score, 1)
                    }
                    overall_score += param_score
                
                overall_score = round(overall_score / num_params, 1)
                
                # Suitability rating
                if overall_score >= 85:
                    suitability = 'Excellent'
                    color = '#28a745'
                elif overall_score >= 70:
                    suitability = 'Good'
                    color = '#17a2b8'
                elif overall_score >= 50:
                    suitability = 'Moderate'
                    color = '#ffc107'
                elif overall_score >= 30:
                    suitability = 'Poor'
                    color = '#fd7e14'
                else:
                    suitability = 'Unsuitable'
                    color = '#dc3545'
                
                crop_info = CROP_INFO.get(crop, {})
                
                # Predict what the model would recommend for these conditions
                input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
                input_scaled = scaler.transform(input_data)
                prediction_encoded = model.predict(input_scaled)
                recommended_crop = label_encoder.inverse_transform(prediction_encoded)[0]
                match = (recommended_crop == crop)
                
                return render_template('result.html',
                    suitability=suitability,
                    suitability_color=color,
                    overall_score=overall_score,
                    deviations=deviations,
                    crop_name=crop,
                    crop_info=crop_info,
                    recommended_crop=recommended_crop,
                    match=match,
                    scenario='suitability')
            else:
                return render_template('result.html',
                    error=f'Crop "{crop}" not found in the database.')
                    
        except Exception as e:
            return render_template('result.html',
                error=f'Suitability check error: {str(e)}')

@app.route('/research_data')
def research_data():
    """
    Scenario 3: Agricultural Research - API endpoint for data
    """
    if df is not None:
        # Crop distribution
        crop_counts = df['label'].value_counts().to_dict()
        
        # Average conditions per crop
        avg_conditions = df.groupby('label').mean().round(2).to_dict(orient='index')
        
        # Convert to serializable format
        crop_names = list(crop_counts.keys())
        crop_counts_list = list(crop_counts.values())
        
        # Nutrient averages
        nutrient_avg = {
            'N': df.groupby('label')['N'].mean().round(1).to_dict(),
            'P': df.groupby('label')['P'].mean().round(1).to_dict(),
            'K': df.groupby('label')['K'].mean().round(1).to_dict()
        }
        
        # Environmental averages
        env_avg = {
            'temperature': df.groupby('label')['temperature'].mean().round(1).to_dict(),
            'humidity': df.groupby('label')['humidity'].mean().round(1).to_dict(),
            'ph': df.groupby('label')['ph'].mean().round(2).to_dict(),
            'rainfall': df.groupby('label')['rainfall'].mean().round(1).to_dict()
        }
        
        return jsonify({
            'success': True,
            'crop_names': crop_names,
            'crop_counts': crop_counts_list,
            'nutrient_avg': nutrient_avg,
            'env_avg': env_avg,
            'total_samples': len(df),
            'num_crops': df['label'].nunique()
        })
    else:
        return jsonify({'success': False, 'error': 'Dataset not loaded'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for crop prediction"""
    try:
        data = request.get_json()
        
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        input_data = np.array([[
            float(data['N']), float(data['P']), float(data['K']),
            float(data['temperature']), float(data['humidity']),
            float(data['ph']), float(data['rainfall'])
        ]])
        
        input_scaled = scaler.transform(input_data)
        prediction_encoded = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        predicted_crop = label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = float(prediction_proba[0][prediction_encoded[0]])
        
        # Top 3 predictions
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        top_3 = [
            {
                'crop': str(label_encoder.inverse_transform([idx])[0]),
                'confidence': round(float(prediction_proba[0][idx] * 100), 2)
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            'success': True,
            'predicted_crop': predicted_crop,
            'confidence': round(confidence * 100, 2),
            'top_3_predictions': top_3
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌾 OptiCrop - Smart Agricultural Production Optimization Engine")
    print("=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)