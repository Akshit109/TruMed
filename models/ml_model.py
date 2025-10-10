import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

SYMPTOM_CATEGORIES = {
    'Skin & Dermatological': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'yellowish_skin',
        'dischromic_patches', 'pus_filled_pimples', 'blackheads', 
        'scurring', 'skin_peeling', 'silver_like_dusting', 
        'small_dents_in_nails', 'inflammatory_nails', 'blister',
        'red_sore_around_nose', 'yellow_crust_ooze', 'brittle_nails'
    ],
    
    'Respiratory System': [
        'continuous_sneezing', 'cough', 'breathlessness', 'phlegm',
        'throat_irritation', 'sinus_pressure', 'runny_nose', 'congestion',
        'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum',
        'patches_in_throat'
    ],
    
    'Digestive & Gastrointestinal': [
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting',
        'indigestion', 'loss_of_appetite', 'constipation', 'abdominal_pain',
        'diarrhoea', 'belly_pain', 'passage_of_gases', 'internal_itching',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'swelling_of_stomach', 'distention_of_abdomen',
        'stomach_bleeding', 'nausea'
    ],
    
    'Cardiovascular & Circulatory': [
        'chest_pain', 'fast_heart_rate', 'palpitations',
        'swollen_blood_vessels', 'prominent_veins_on_calf',
        'swollen_legs'
    ],
    
    'Neurological & Mental Health': [
        'headache', 'dizziness', 'loss_of_balance', 'unsteadiness',
        'spinning_movements', 'weakness_of_one_body_side', 'loss_of_smell',
        'altered_sensorium', 'lack_of_concentration', 'visual_disturbances',
        'blurred_and_distorted_vision', 'anxiety', 'mood_swings',
        'depression', 'irritability', 'slurred_speech', 'coma'
    ],
    
    'Musculoskeletal System': [
        'joint_pain', 'muscle_wasting', 'back_pain', 'neck_pain',
        'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
        'swelling_joints', 'movement_stiffness', 'muscle_pain',
        'weakness_in_limbs', 'painful_walking', 'cramps', 'bruising'
    ],
    
    'Urinary System': [
        'burning_micturition', 'spotting_urination', 'bladder_discomfort',
        'foul_smell_of_urine', 'continuous_feel_of_urine', 'dark_urine',
        'yellow_urine', 'polyuria'
    ],
    
    'Systemic & General': [
        'fatigue', 'weight_gain', 'weight_loss', 'restlessness',
        'lethargy', 'high_fever', 'mild_fever', 'sweating',
        'dehydration', 'chills', 'shivering', 'malaise',
        'obesity', 'excessive_hunger', 'increased_appetite',
        'cold_hands_and_feets'
    ],
    
    'Eyes & Vision': [
        'sunken_eyes', 'pain_behind_the_eyes', 'yellowing_of_eyes',
        'redness_of_eyes', 'watering_from_eyes', 'puffy_face_and_eyes'
    ],
    
    'Endocrine & Metabolic': [
        'irregular_sugar_level', 'enlarged_thyroid', 'swollen_extremeties'
    ],
    
    'Liver & Hepatic': [
        'acute_liver_failure', 'fluid_overload', 'swelled_lymph_nodes',
        'history_of_alcohol_consumption'
    ],
    
    'Reproductive & Sexual': [
        'abnormal_menstruation', 'extra_marital_contacts',
        'drying_and_tingling_lips'
    ],
    
    'Blood & Immune System': [
        'red_spots_over_body', 'receiving_blood_transfusion',
        'receiving_unsterile_injections'
    ],
    
    'Other Medical History': [
        'family_history', 'toxic_look_(typhos)'
    ]
}

# ====================================================================
# DISEASE-SYMPTOM MAPPING - Realistic medical associations
# ====================================================================

DISEASE_SYMPTOM_MAP = {
    'Fungal infection': {
        'primary': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'],
        'secondary': ['skin_peeling', 'scurring', 'small_dents_in_nails']
    },
    'Allergy': {
        'primary': ['continuous_sneezing', 'shivering', 'skin_rash', 'watering_from_eyes'],
        'secondary': ['runny_nose', 'congestion', 'throat_irritation', 'redness_of_eyes']
    },
    'GERD': {
        'primary': ['acidity', 'ulcers_on_tongue', 'vomiting', 'chest_pain'],
        'secondary': ['stomach_pain', 'indigestion', 'cough', 'throat_irritation']
    },
    'Chronic cholestasis': {
        'primary': ['itching', 'vomiting', 'yellowish_skin', 'nausea'],
        'secondary': ['loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes']
    },
    'Drug Reaction': {
        'primary': ['itching', 'skin_rash', 'vomiting', 'fatigue'],
        'secondary': ['nausea', 'loss_of_appetite', 'stomach_pain']
    },
    'Peptic ulcer diseae': {
        'primary': ['vomiting', 'loss_of_appetite', 'abdominal_pain', 'stomach_pain'],
        'secondary': ['nausea', 'passage_of_gases', 'indigestion', 'internal_itching']
    },
    'AIDS': {
        'primary': ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'],
        'secondary': ['fatigue', 'weight_loss', 'cough', 'malaise']
    },
    'Diabetes': {
        'primary': ['fatigue', 'weight_loss', 'restlessness', 'lethargy'],
        'secondary': ['irregular_sugar_level', 'blurred_and_distorted_vision', 'obesity', 'excessive_hunger']
    },
    'Gastroenteritis': {
        'primary': ['vomiting', 'sunken_eyes', 'dehydration', 'diarrhoea'],
        'secondary': ['nausea', 'stomach_pain', 'headache', 'loss_of_appetite']
    },
    'Bronchial Asthma': {
        'primary': ['fatigue', 'cough', 'high_fever', 'breathlessness'],
        'secondary': ['mucoid_sputum', 'chest_pain', 'wheezing', 'malaise']
    },
    'Hypertension': {
        'primary': ['headache', 'chest_pain', 'dizziness', 'loss_of_balance'],
        'secondary': ['lack_of_concentration', 'blurred_and_distorted_vision']
    },
    'Migraine': {
        'primary': ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision'],
        'secondary': ['excessive_hunger', 'nausea', 'depression', 'irritability']
    },
    'Cervical spondylosis': {
        'primary': ['back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness'],
        'secondary': ['loss_of_balance', 'unsteadiness', 'headache']
    },
    'Paralysis (brain hemorrhage)': {
        'primary': ['vomiting', 'headache', 'weakness_of_one_body_side', 'altered_sensorium'],
        'secondary': ['slurred_speech', 'loss_of_balance', 'dizziness']
    },
    'Jaundice': {
        'primary': ['itching', 'vomiting', 'fatigue', 'weight_loss'],
        'secondary': ['high_fever', 'yellowish_skin', 'dark_urine', 'yellowing_of_eyes']
    },
    'Malaria': {
        'primary': ['chills', 'vomiting', 'high_fever', 'sweating'],
        'secondary': ['headache', 'nausea', 'diarrhoea', 'muscle_pain']
    },
    'Chicken pox': {
        'primary': ['itching', 'skin_rash', 'fatigue', 'lethargy'],
        'secondary': ['high_fever', 'headache', 'loss_of_appetite', 'mild_fever']
    },
    'Dengue': {
        'primary': ['skin_rash', 'chills', 'joint_pain', 'vomiting'],
        'secondary': ['fatigue', 'high_fever', 'headache', 'nausea', 'muscle_pain']
    },
    'Typhoid': {
        'primary': ['chills', 'vomiting', 'fatigue', 'high_fever'],
        'secondary': ['headache', 'nausea', 'constipation', 'abdominal_pain', 'toxic_look_(typhos)']
    },
    'Hepatitis A': {
        'primary': ['joint_pain', 'vomiting', 'yellowish_skin', 'dark_urine'],
        'secondary': ['nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes']
    },
    'Hepatitis B': {
        'primary': ['itching', 'fatigue', 'lethargy', 'yellowish_skin'],
        'secondary': ['dark_urine', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes']
    },
    'Hepatitis C': {
        'primary': ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite'],
        'secondary': ['yellowing_of_eyes', 'family_history', 'dark_urine']
    },
    'Hepatitis D': {
        'primary': ['joint_pain', 'vomiting', 'fatigue', 'high_fever'],
        'secondary': ['yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite']
    },
    'Hepatitis E': {
        'primary': ['joint_pain', 'vomiting', 'fatigue', 'high_fever'],
        'secondary': ['yellowish_skin', 'dark_urine', 'nausea', 'acute_liver_failure']
    },
    'Alcoholic hepatitis': {
        'primary': ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach'],
        'secondary': ['distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload']
    },
    'Tuberculosis': {
        'primary': ['chills', 'vomiting', 'fatigue', 'weight_loss'],
        'secondary': ['cough', 'high_fever', 'breathlessness', 'sweating', 'blood_in_sputum']
    },
    'Common Cold': {
        'primary': ['continuous_sneezing', 'chills', 'fatigue', 'cough'],
        'secondary': ['high_fever', 'headache', 'swelled_lymph_nodes', 'malaise', 'runny_nose']
    },
    'Pneumonia': {
        'primary': ['chills', 'fatigue', 'cough', 'high_fever'],
        'secondary': ['breathlessness', 'sweating', 'malaise', 'phlegm', 'chest_pain']
    },
    'Dimorphic hemmorhoids(piles)': {
        'primary': ['constipation', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool'],
        'secondary': ['irritation_in_anus', 'internal_itching']
    },
    'Heart attack': {
        'primary': ['vomiting', 'chest_pain', 'breathlessness', 'sweating'],
        'secondary': ['nausea', 'fast_heart_rate', 'dizziness']
    },
    'Varicose veins': {
        'primary': ['fatigue', 'cramps', 'bruising', 'obesity'],
        'secondary': ['swollen_legs', 'swollen_blood_vessels', 'prominent_veins_on_calf']
    },
    'Hypothyroidism': {
        'primary': ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings'],
        'secondary': ['lethargy', 'dizziness', 'puffy_face_and_eyes', 'enlarged_thyroid']
    },
    'Hyperthyroidism': {
        'primary': ['fatigue', 'mood_swings', 'weight_loss', 'restlessness'],
        'secondary': ['sweating', 'diarrhoea', 'fast_heart_rate', 'excessive_hunger']
    },
    'Hypoglycemia': {
        'primary': ['vomiting', 'fatigue', 'anxiety', 'sweating'],
        'secondary': ['headache', 'nausea', 'blurred_and_distorted_vision', 'slurred_speech']
    },
    'Osteoarthristis': {
        'primary': ['joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain'],
        'secondary': ['painful_walking', 'swelling_joints', 'movement_stiffness']
    },
    'Arthritis': {
        'primary': ['muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness'],
        'secondary': ['painful_walking', 'joint_pain', 'muscle_pain']
    },
    '(vertigo) Paroymsal  Positional Vertigo': {
        'primary': ['vomiting', 'headache', 'nausea', 'spinning_movements'],
        'secondary': ['loss_of_balance', 'unsteadiness']
    },
    'Acne': {
        'primary': ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],
        'secondary': ['skin_peeling', 'inflammatory_nails']
    },
    'Urinary tract infection': {
        'primary': ['burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine'],
        'secondary': ['dark_urine', 'nausea', 'abdominal_pain']
    },
    'Psoriasis': {
        'primary': ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting'],
        'secondary': ['small_dents_in_nails', 'inflammatory_nails', 'scurring']
    },
    'Impetigo': {
        'primary': ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose'],
        'secondary': ['yellow_crust_ooze', 'itching']
    }
}

# ====================================================================
# PREDICTOR CLASS - Main Interface with Category Support
# ====================================================================

class DiseasePredictor:
    """
    Main predictor class that handles disease prediction with explainability.
    Compliant with GDPR Article 22 and Indian IT Rules.
    NOW WITH SYMPTOM CATEGORIZATION FOR BETTER UX
    """
    
    def __init__(self):
        """Initialize the predictor with comprehensive symptom list"""
        self.symptoms_list = [
            'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
            'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
            'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
            'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
            'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
            'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
            'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
            'dehydration', 'indigestion', 'headache', 'yellowish_skin',
            'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
            'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
            'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
            'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
            'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
            'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
            'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
            'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
            'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
            'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
            'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
            'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
            'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
            'spinning_movements', 'loss_of_balance', 'unsteadiness',
            'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
            'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases',
            'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
            'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
            'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
            'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
            'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
            'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
            'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
            'blood_in_sputum', 'prominent_veins_on_calf',
            'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
            'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
            'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
        ]
        
        self.symptom_categories = SYMPTOM_CATEGORIES
        self.disease_symptom_map = DISEASE_SYMPTOM_MAP
        self.model = None
        self.scaler = None
        self.le_target = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = None
        self.disease_list = []
        self.is_trained = False
        
        # Try to load pre-trained model
        self._load_model()
    
    def get_symptoms_by_category(self, category_name=None):
        """
        Get symptoms organized by category
        
        Args:
            category_name: Specific category name, or None to get all
            
        Returns:
            Dictionary of categories with symptoms, or list of symptoms for specific category
        """
        if category_name:
            return self.symptom_categories.get(category_name, [])
        else:
            # Return all categories with formatted symptom names
            formatted_categories = {}
            for category, symptoms in self.symptom_categories.items():
                formatted_categories[category] = [
                    {
                        'value': symptom,
                        'label': symptom.replace('_', ' ').title()
                    }
                    for symptom in symptoms
                ]
            return formatted_categories
    
    def get_all_categories(self):
        """Get list of all category names"""
        return list(self.symptom_categories.keys())
    
    def search_symptoms(self, query):
        """
        Search for symptoms matching a query string
        
        Args:
            query: Search term
            
        Returns:
            List of matching symptoms with their categories
        """
        query_lower = query.lower()
        results = []
        
        for category, symptoms in self.symptom_categories.items():
            for symptom in symptoms:
                if query_lower in symptom.replace('_', ' ').lower():
                    results.append({
                        'symptom': symptom,
                        'label': symptom.replace('_', ' ').title(),
                        'category': category
                    })
        
        return results
    
    def _load_model(self):
        """Load pre-trained model if available"""
        model_path = os.path.join('models', 'disease_prediction_model.pkl')
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
                
                self.model = model_package.get('model')
                self.scaler = model_package.get('scaler')
                self.le_target = model_package.get('le_target')
                self.shap_explainer = model_package.get('shap_explainer')
                self.lime_explainer = model_package.get('lime_explainer')
                self.feature_names = model_package.get('feature_names', self.symptoms_list)
                
                if self.le_target:
                    self.disease_list = list(self.le_target.classes_)
                
                self.is_trained = True
                print(f"✓ Pre-trained model loaded successfully")
                print(f"✓ Model accuracy: {model_package.get('accuracy', 0)*100:.2f}%")
            except Exception as e:
                print(f"⚠ Could not load pre-trained model: {e}")
                self._create_fallback_model()
        else:
            print("⚠ No pre-trained model found. Creating fallback model...")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a model with realistic disease-symptom relationships"""
        print("Creating realistic training data...")
        
        diseases = list(DISEASE_SYMPTOM_MAP.keys())
        
        # Create realistic dataset
        np.random.seed(42)
        n_samples_per_disease = 150  # Increased samples
        
        X_data = []
        y_data = []
        
        # Create symptom index mapping
        symptom_to_idx = {symptom: idx for idx, symptom in enumerate(self.symptoms_list)}
        
        for disease_idx, disease in enumerate(diseases):
            symptom_profile = DISEASE_SYMPTOM_MAP[disease]
            primary_symptoms = symptom_profile['primary']
            secondary_symptoms = symptom_profile.get('secondary', [])
            
            for _ in range(n_samples_per_disease):
                # Initialize empty symptom vector
                symptom_vector = np.zeros(len(self.symptoms_list))
                
                # Add primary symptoms (high probability: 80-100%)
                for symptom in primary_symptoms:
                    if symptom in symptom_to_idx:
                        if np.random.random() > 0.15:  # 85% chance
                            symptom_vector[symptom_to_idx[symptom]] = 1
                
                # Add secondary symptoms (moderate probability: 40-70%)
                for symptom in secondary_symptoms:
                    if symptom in symptom_to_idx:
                        if np.random.random() > 0.4:  # 60% chance
                            symptom_vector[symptom_to_idx[symptom]] = 1
                
                # Add some noise symptoms (5-10% chance for random symptoms)
                noise_indices = np.random.choice(
                    len(self.symptoms_list), 
                    size=np.random.randint(0, 3),
                    replace=False
                )
                for idx in noise_indices:
                    if np.random.random() > 0.92:  # 8% chance
                        symptom_vector[idx] = 1
                
                X_data.append(symptom_vector)
                y_data.append(disease_idx)
        
        X = pd.DataFrame(X_data, columns=self.symptoms_list)
        y = np.array(y_data)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        print("Training XGBoost model...")
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ Model trained with {accuracy*100:.2f}% accuracy")
        
        # Create label encoder for diseases
        self.le_target = LabelEncoder()
        self.le_target.classes_ = np.array(diseases)
        self.disease_list = diseases
        
        # Create explainers
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_scaled,
            feature_names=self.symptoms_list,
            class_names=diseases,
            mode='classification'
        )
        
        self.feature_names = self.symptoms_list
        self.is_trained = True
        
        # Save model
        self._save_model(accuracy)
    
    def _save_model(self, accuracy):
        """Save trained model"""
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'le_target': self.le_target,
            'shap_explainer': self.shap_explainer,
            'lime_explainer': self.lime_explainer,
            'feature_names': self.feature_names,
            'accuracy': accuracy
        }
        
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'disease_prediction_model.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✓ Model saved to {model_path}")
    
    def predict(self, symptoms_dict):
        """
        Make prediction with explainability
        
        Args:
            symptoms_dict: Dictionary with symptom names as keys and boolean values
            
        Returns:
            Dictionary with prediction results and explainability data
        """
        if not self.is_trained:
            return {
                'disease': 'Model Not Trained',
                'confidence': 0.0,
                'risk_level': 'Unknown',
                'active_symptoms': [],
                'contributing_factors': [],
                'suggestions': ['Please train the model first']
            }
        
        # Convert symptoms dict to feature vector
        feature_vector = []
        active_symptoms = []
        
        for symptom in self.symptoms_list:
            if symptoms_dict.get(symptom, False):
                feature_vector.append(1)
                active_symptoms.append(symptom.replace('_', ' ').title())
            else:
                feature_vector.append(0)
        
        # Scale features
        X_input = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X_input)
        
        # Make prediction
        prediction_idx = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = float(max(probabilities)) * 100
        
        # Get disease name
        if self.le_target:
            disease_name = self.le_target.inverse_transform([prediction_idx])[0]
        else:
            disease_name = f"Disease_{prediction_idx}"
        
        # Calculate risk level based on confidence
        if confidence >= 75:
            risk_level = 'High'
        elif confidence >= 50:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
        
        # Get SHAP values for explainability
        contributing_factors = []
        try:
            shap_values = self.shap_explainer.shap_values(X_scaled)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[prediction_idx][0]
            else:
                shap_vals = shap_values[0]
            
            # Get top contributing features
            feature_importance = []
            for i, val in enumerate(shap_vals):
                if feature_vector[i] == 1:  # Only active symptoms
                    feature_importance.append((self.symptoms_list[i], abs(val)))
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Format for output
            for symptom, importance in feature_importance[:10]:
                contributing_factors.append({
                    'symptom': symptom.replace('_', ' ').title(),
                    'importance': float(importance),
                    'impact': 'High' if importance > 0.1 else 'Moderate' if importance > 0.05 else 'Low'
                })
        except Exception as e:
            print(f"SHAP calculation error: {e}")
        
        # Generate suggestions
        suggestions = self._generate_suggestions(disease_name, risk_level, active_symptoms)
        
        return {
            'disease': disease_name,
            'confidence': confidence,
            'risk_level': risk_level,
            'active_symptoms': active_symptoms,
            'contributing_factors': contributing_factors,
            'suggestions': suggestions,
            'all_probabilities': {
                self.disease_list[i]: float(prob) * 100 
                for i, prob in enumerate(probabilities)
            }
        }
    
    def _generate_suggestions(self, disease, risk_level, symptoms):
        """Generate medical suggestions based on prediction"""
        suggestions = []
        
        if risk_level == 'High':
            suggestions.append("⚠️ High confidence prediction - Please consult a healthcare professional immediately")
            suggestions.append("📋 Get a comprehensive medical examination")
        elif risk_level == 'Moderate':
            suggestions.append("⚕️ Moderate confidence - Recommend medical consultation")
            suggestions.append("📊 Consider getting relevant diagnostic tests")
        else:
            suggestions.append("ℹ️ Low confidence - Monitor symptoms and consult if they persist")
        
        # General health advice
        suggestions.append("💊 Do not self-medicate without professional advice")
        suggestions.append("📝 Keep a record of your symptoms and their progression")
        suggestions.append("🏥 Visit a certified healthcare provider for accurate diagnosis")
        suggestions.append("💪 Maintain a healthy lifestyle with proper diet and exercise")
        
        # Disease-specific suggestions
        disease_lower = disease.lower()
        if 'infection' in disease_lower or 'fever' in disease_lower:
            suggestions.append("🌡️ Monitor your body temperature regularly")
            suggestions.append("💧 Stay well-hydrated and get adequate rest")
        
        if 'diabetes' in disease_lower:
            suggestions.append("🍎 Monitor blood sugar levels and follow a diabetic diet")
        
        if 'heart' in disease_lower or 'hypertension' in disease_lower:
            suggestions.append("❤️ Monitor blood pressure and avoid high-sodium foods")
        
        if 'asthma' in disease_lower or 'breathlessness' in ' '.join(symptoms).lower():
            suggestions.append("🫁 Keep rescue inhaler accessible if prescribed")
        
        return suggestions

# ====================================================================
# INITIALIZE GLOBAL PREDICTOR INSTANCE
# ====================================================================

# Create global predictor instance for Flask app
predictor = DiseasePredictor()

print("=" * 80)
print("DISEASE PREDICTION MODEL INITIALIZED")
print(f"Total Symptoms: {len(predictor.symptoms_list)}")
print(f"Total Diseases: {len(predictor.disease_list)}")
print(f"Total Categories: {len(predictor.symptom_categories)}")
print(f"Model Status: {'✓ Trained' if predictor.is_trained else '✗ Not Trained'}")
print(f"Explainability: SHAP + LIME Enabled")
print(f"Compliance: GDPR Article 22 + Indian IT Rules")
print("=" * 80)
