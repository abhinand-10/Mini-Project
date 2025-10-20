from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///taxapp.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    calculations = db.relationship('Calculation', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Calculation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    calculation_type = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

# Authentication decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# --- Model Loading (Your existing code) ---
DEFAULT_TDS_FEATURES = [
    'payment_amount', 'tds_rate', 'tds_deducted', 'tds_deposited', 
    'day_of_week', 'month', 'is_weekend', 'section_code_194A', 
    'section_code_194C', 'section_code_194H', 'section_code_194I', 
    'nature_of_payment_Commission', 'nature_of_payment_Contractor', 
    'nature_of_payment_Rent'
]

DEFAULT_TAX_PREDICTION_FEATURES = ['current_income']

# Load GST Fraud Detection Model
try:
    gst_fraud_model_path = os.path.join("models", "gst_fraud_model.pkl")
    if os.path.exists(gst_fraud_model_path):
        with open(gst_fraud_model_path, "rb") as f:
            gst_fraud_model = pickle.load(f)
        print("✅ GST fraud model loaded successfully (via pickle).")
    else:
        gst_fraud_model = None
        print("⚠️ GST fraud model file not found.")
except Exception as e:
    gst_fraud_model = None
    print(f"⚠️ Could not load GST fraud model: {e}")

# Load TDS Fraud Detection Model (simplified for now)
try:
    tds_model_path = os.path.join("models", "tds_fraud_model.pkl")
    if os.path.exists(tds_model_path):
        with open(tds_model_path, "rb") as f:
            tds_model_data = pickle.load(f)
        tds_model = tds_model_data.get('model') if isinstance(tds_model_data, dict) else tds_model_data
        tds_features = DEFAULT_TDS_FEATURES
        print("✅ TDS fraud model loaded successfully.")
    else:
        tds_model = None
        tds_features = []
        print("⚠️ TDS fraud model file not found.")
except Exception as e:
    tds_model = None
    tds_features = []
    print(f"⚠️ Could not load TDS fraud model: {e}")

# Load Tax Prediction Model
try:
    tax_model_path = os.path.join("models", "tax_prediction_model.pkl")
    if os.path.exists(tax_model_path):
        tax_prediction_model = pickle.load(open(tax_model_path, "rb"))
        print("✅ Tax prediction model loaded successfully.")
    else:
        tax_prediction_model = None
        print("⚠️ Tax prediction model file not found.")
except Exception as e:
    tax_prediction_model = None
    print(f"⚠️ Could not load tax prediction model: {e}")

# --- AUTHENTICATION ROUTES ---
@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template("signup.html")
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template("signup.html")

        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login_page'))
        
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# --- SAVE CALCULATION ROUTE ---
@app.route("/save_calculation", methods=["POST"])
@login_required
def save_calculation():
    try:
        data = request.get_json()
        calculation_type = data.get('type')
        input_data = data.get('input_data')
        result = data.get('result')
        
        calculation = Calculation(
            user_id=session['user_id'],
            calculation_type=calculation_type,
            input_data=json.dumps(input_data),
            result=json.dumps(result)
        )
        
        db.session.add(calculation)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Calculation saved successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# --- DASHBOARD ROUTE ---
@app.route("/dashboard")
@login_required
def dashboard():
    user_calculations = Calculation.query.filter_by(
        user_id=session['user_id']
    ).order_by(Calculation.timestamp.desc()).all()
    
    # Parse JSON data for template
    calculations_data = []
    for calc in user_calculations:
        try:
            calculations_data.append({
                'id': calc.id,
                'type': calc.calculation_type,
                'input_data': json.loads(calc.input_data),
                'result': json.loads(calc.result),
                'timestamp': calc.timestamp
            })
        except json.JSONDecodeError:
            continue
    
    return render_template("dashboard.html", calculations=calculations_data)

# --- CORE ROUTES ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/gst", methods=["GET"])
def gst_page():
    return render_template("gst.html")

@app.route("/favicon.ico")
def favicon():
    return ""

# --- GST FRAUD DETECTION ---
@app.route("/detect_fraud", methods=["POST"])
def detect_fraud():
    if gst_fraud_model is None:
        return render_template("gst.html", fraud_error="⚠️ GST fraud model not loaded. Please train/load the model.")

    try:
        # Get form data
        reported_turnover = float(request.form["reported_turnover"])
        eway_total_value = float(request.form["eway_total_value"])
        num_eway_bills = float(request.form["num_eway_bills"])
        invoice_count = float(request.form["invoice_count"])
        gst_paid = float(request.form["gst_paid"])
        gst_rate_applied = float(request.form["gst_rate_applied"])
        num_suppliers = float(request.form["num_suppliers"])
        num_customers = float(request.form["num_customers"])
        avg_invoice_value = float(request.form["avg_invoice_value"])
        gstr1_vs_gstr3b_diff = float(request.form["gstr1_vs_gstr3b_diff"])
        sudden_turnover_jump = int(request.form["sudden_turnover_jump"])
        registration_age_months = float(request.form["registration_age_months"])
        
        # Create a pandas DataFrame for the model
        input_data = pd.DataFrame([[
            reported_turnover, eway_total_value, num_eway_bills, invoice_count,
            gst_paid, gst_rate_applied, num_suppliers, num_customers,
            avg_invoice_value, gstr1_vs_gstr3b_diff, sudden_turnover_jump,
            registration_age_months
        ]], columns=[
            'reported_turnover', 'eway_total_value', 'num_eway_bills', 'invoice_count',
            'gst_paid', 'gst_rate_applied', 'num_suppliers', 'num_customers',
            'avg_invoice_value', 'gstr1_vs_gstr3b_diff', 'sudden_turnover_jump',
            'registration_age_months'
        ])

        prediction = gst_fraud_model.predict(input_data)[0]

        if prediction == 1:
            fraud_result = "⚠️ Suspicious GST activity detected — Possible Fraud!"
        else:
            fraud_result = "✅ GST record looks normal."

        return render_template("gst.html", fraud_result=fraud_result)

    except Exception as e:
        return render_template("gst.html", fraud_error=f"❌ Error: {str(e)}")

# --- TDS FRAUD DETECTION ---
@app.route("/tds", methods=["GET", "POST"])
@app.route("/tds_check", methods=["POST"])
def tds_page():
    if request.method == "POST":
        try:
            # Check if model is loaded and has features
            if tds_model is None or not tds_features:
                return jsonify({"error": "TDS fraud model not loaded or features list is empty."}), 500
                
            # Get form data from the POST request
            data = {
                'deductor_pan': request.form.get('deductor_pan'),
                'deductee_pan': request.form.get('deductee_pan'),
                'payment_amount': float(request.form.get('payment_amount')),
                'tds_rate': float(request.form.get('tds_rate')),
                'tds_deducted': float(request.form.get('tds_deducted')),
                'tds_deposited': float(request.form.get('tds_deposited')),
                'section_code': request.form.get('section_code'),
                'nature_of_payment': request.form.get('nature_of_payment'),
                'date_of_payment': request.form.get('date_of_payment')
            }
            
            # Create a DataFrame for preprocessing
            input_df = pd.DataFrame([data])
            
            # --- Preprocessing to match training script ---
            # Date features
            input_df['date_of_payment'] = pd.to_datetime(input_df['date_of_payment'])
            input_df['day_of_week'] = input_df['date_of_payment'].dt.dayofweek
            input_df['month'] = input_df['date_of_payment'].dt.month
            input_df['is_weekend'] = input_df['day_of_week'].isin([5, 6]).astype(int)

            # Create a DataFrame with all expected features, filled with zeros
            processed_df = pd.DataFrame(index=[0], columns=tds_features, data=0)

            # Copy numerical values from input
            processed_df.loc[0, 'payment_amount'] = data['payment_amount']
            processed_df.loc[0, 'tds_rate'] = data['tds_rate']
            processed_df.loc[0, 'tds_deducted'] = data['tds_deducted']
            processed_df.loc[0, 'tds_deposited'] = data['tds_deposited']
            processed_df.loc[0, 'day_of_week'] = input_df.loc[0, 'day_of_week']
            processed_df.loc[0, 'month'] = input_df.loc[0, 'month']
            processed_df.loc[0, 'is_weekend'] = input_df.loc[0, 'is_weekend']

            # Set one-hot encoded columns based on input
            section_col = f"section_code_{data['section_code']}"
            payment_col = f"nature_of_payment_{data['nature_of_payment']}"
            
            if section_col in processed_df.columns:
                processed_df.loc[0, section_col] = 1
            if payment_col in processed_df.columns:
                processed_df.loc[0, payment_col] = 1
            
            # Ensure the feature order is correct for the model
            final_input_df = processed_df[tds_features]

            # Make the prediction
            prediction = tds_model.predict(final_input_df)[0]
            fraud_result = "Fraudulent Transaction ⚠️" if prediction == 1 else "Normal Transaction ✅"

            return jsonify({"fraudCheck": fraud_result})

        except ValueError:
             return jsonify({"error": "Please ensure all numerical fields have valid inputs."}), 400
        except Exception as e:
            return jsonify({"error": f"TDS prediction error: {str(e)}"}), 500

    return render_template("tds.html")

# --- INCOME TAX CALCULATIONS ---
def compute_old_regime_tax(taxable_income, age_category):
    tax = 0
    
    if age_category == "general":  # Below 60
        slab1, slab2 = 250000, 500000
    elif age_category == "senior":  # 60-80
        slab1, slab2 = 300000, 500000
    else:  # super senior (Above 80)
        slab1, slab2 = 500000, 1000000

    if taxable_income <= slab1:
        tax = 0
    elif taxable_income <= slab2:
        tax = (taxable_income - slab1) * 0.05
    elif taxable_income <= 1000000:
        tax = (slab2 - slab1) * 0.05 + (taxable_income - slab2) * 0.20
    else:
        tax = (slab2 - slab1) * 0.05 + (1000000 - slab2) * 0.20 + (taxable_income - 1000000) * 0.30

    return tax

def compute_new_regime_tax(taxable_income):
    tax = 0
    slabs = [300000, 600000, 900000, 1200000, 1500000]
    rates = [0.05, 0.10, 0.15, 0.20, 0.30]
    
    prev_limit = 0
    for i, limit in enumerate(slabs):
        if taxable_income > limit:
            tax += (limit - prev_limit) * rates[i]
            prev_limit = limit
        else:
            tax += (taxable_income - prev_limit) * rates[i]
            return tax
            
    if taxable_income > slabs[-1]:
        tax += (taxable_income - slabs[-1]) * 0.30
    
    return tax

def get_float_value(field_name):
    value = request.form.get(field_name, "").strip()
    return float(value) if value else 0.0

@app.route("/income_tax", methods=["GET", "POST"])
def income_tax():
    old_tax, new_tax, best_regime = None, None, None
    if request.method == "POST":
        try:
            gross_salary = get_float_value("gross_salary")
            income_other = get_float_value("income_other")
            income_interest = get_float_value("income_interest")
            rental_income = get_float_value("rental_income")
            home_loan_self = get_float_value("home_loan_self")
            home_loan_letout = get_float_value("home_loan_letout")
            deduction_80C = get_float_value("deduction_80C")
            contribution = get_float_value("contribution")
            medical = get_float_value("medical")
            donation = get_float_value("donation")
            edu_loan = get_float_value("edu_loan")
            savings_interest = get_float_value("savings_interest")
            basic_salary = get_float_value("basic_salary")
            da_salary = get_float_value("da_salary")
            hra_received = get_float_value("hra_received")
            rent_paid = get_float_value("rent_paid")
            age_category = request.form.get("age_category", "general")

            gross_income = gross_salary + income_other + income_interest + rental_income
            deductions = deduction_80C + contribution + medical + donation + edu_loan + savings_interest
            
            # Apply home loan interest
            deductions += home_loan_self
            gross_income -= home_loan_letout
            
            taxable_old = max(0, gross_income - deductions)
            taxable_new = max(0, gross_income)

            old_tax = compute_old_regime_tax(taxable_old, age_category)
            new_tax = compute_new_regime_tax(taxable_new)
            best_regime = "Old Regime" if old_tax < new_tax else "New Regime"
        except Exception as e:
            print(f"Income tax calculation error: {e}")

    return render_template("income_tax.html", old_tax=old_tax, new_tax=new_tax, best_regime=best_regime)

# --- TAX PREDICTION ---
@app.route("/tax_prediction", methods=["GET"])
def tax_prediction_page():
    return render_template("tax_prediction.html")

@app.route("/predict_tax", methods=["POST"])
def predict_tax():
    try:
        # Check if the model is loaded
        if tax_prediction_model is None:
            return jsonify({"error": "Tax prediction model not loaded. Please train the model first."}), 500

        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data received."}), 400
            
        current_income = data.get("current_income")
        
        # Validate input
        if current_income is None:
            return jsonify({"error": "Current income is a required field."}), 400
        
        current_income = float(current_income)

        # Prepare the input for the model
        input_features = np.array([[current_income]])
        
        # Make the prediction
        predicted_tax = tax_prediction_model.predict(input_features)[0]
        predicted_tax = max(0, predicted_tax)
        
        return jsonify({
            "predicted_tax": round(predicted_tax, 2)
        })

    except ValueError:
        return jsonify({"error": "Invalid numerical value for income."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    app.run(debug=True)