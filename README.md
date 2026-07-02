# OptiCrop-Agricultural-Production

📖 Usage Guide
1️⃣ Crop Recommendation
Navigate to Crop Recommendation → Enter N, P, K, Temperature, Humidity, pH, Rainfall → Click Recommend Crop → View result with crop name and confidence.

2️⃣ Suitability Assessment
Navigate to Suitability Assessment → Select a crop from dropdown → Enter soil/weather parameters → Click Check Suitability → View detailed compatibility report.

3️⃣ Research Analytics
Navigate to Research & Analytics → Browse through visualizations and statistics → Use filters to explore specific crops or parameter ranges.

📊 Model Performance
Algorithm Comparison


Algorithm	Accuracy	Precision	Recall	F1-Score
Random Forest (Tuned) ✅	99.55%	0.996	0.996	0.996
Random Forest (Default)	99.32%	0.993	0.993	0.993
Decision Tree	98.64%	0.986	0.986	0.987
XGBoost	98.41%	0.984	0.984	0.984
SVM (RBF)	97.88%	0.979	0.979	0.979
KNN (k=5)	97.50%	0.975	0.975	0.975
Logistic Regression	95.00%	0.952	0.950	0.949
Best Model: Tuned Random Forest
python



RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
Feature Importance


Feature	Importance
Potassium (K)	21.3%
Phosphorous (P)	18.7%
Nitrogen (N)	17.5%
Rainfall	15.2%
Temperature	11.8%
pH	8.6%
Humidity	6.9%
📁 Project Structure



D:\OptiCrop\
├── app.py                          # Flask application server
├── model_training.py               # ML model training script
├── model_training.ipynb            # Jupyter notebook (EDA + training)
├── generate_documentation.py       # PDF documentation generator
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── templates/                      # HTML templates
│   ├── base.html                   # Base layout template
│   ├── index.html                  # Home page
│   ├── recommend.html              # Crop recommendation form
│   ├── suitability.html            # Suitability assessment form
│   ├── research.html               # Research & analytics page
│   └── result.html                 # Results display page
│
├── static/                         # Static assets
│   ├── style.css                   # Custom styles
│   ├── crop_distribution.png       # EDA visualization
│   ├── crop_distribution.html      # Interactive chart
│   ├── feature_distributions.png   # Feature analysis plot
│   ├── correlation_heatmap.png     # Correlation matrix
│   ├── feature_importance.png      # Feature importance chart
│   └── model_comparison.png        # Model accuracy comparison
│
├── models/                         # Trained model files
│   ├── crop_recommendation_model.pkl   # Random Forest model
│   ├── scaler.pkl                      # Feature scaler
│   └── label_encoder.pkl               # Crop type encoder
│
├── Crop_recommendation.csv         # Dataset
│
├── OptiCrop_Project_Documentation.pdf  # Generated documentation
└── venv/                           # Python virtual environment
📸 Screenshots
Home Page
Landing page with hero section, feature cards, crop grid, and model performance chart

Crop Recommendation
Input form with 7 parameters → Result page showing recommended crop and confidence

Suitability Assessment
Dropdown crop selector + input fields → Suitability percentage + detailed report

Research & Analytics
Interactive charts, data tables, crop filtering, and summary statistics

💡 Future Enhancements
🌐 Deploy to production (AWS/GCP/Azure) with Docker containerization
📱 Mobile app with React Native / Flutter
🤖 Real-time IoT sensor integration for automated data collection
🗺️ GIS mapping for location-based recommendations
🌦️ Weather API integration for real-time forecasting
💰 Cost-benefit analysis — estimated profit per crop
📅 Seasonal recommendations based on planting calendars
🧪 Fertilizer recommendation module
🌍 Multi-language support
👥 Project Done By


Role	Name	Email
⭐ Team Lead	Cherukuri Divya Bhanu	divyabhanucherukuri@gmail.com
👤 Member	Polavarapu Gurumurthy	gurumurthypolavarapu@gmail.com
👤 Member	Meda Harika	medahari036@gmail.com
👤 Member	Neelam Venkata Satya Sri	satyaneelam2006@gmail.com
👤 Member	Yakkala Naga Venkata Bhanu Tejaswini	tejayakkalla@gmail.com
📄 License
This project is developed for educational and research purposes under the MIT License.

🌾 OptiCrop — Growing Smarter, Harvesting Better 🌾

Built with ❤️ for the future of agriculture
