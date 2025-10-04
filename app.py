import os
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, flash, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib
import numpy as np
from datetime import datetime
import io
import base64
from models import db, UploadHistory, EmployeeData, ModelMetrics, PerformanceStats, AppSettings

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "hr_performance_eval_secret_key")

# Database configuration
database_url = os.environ.get("DATABASE_URL")
if database_url:
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Fallback to SQLite for development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hr_analytics.db'
    
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
CHARTS_FOLDER = 'static/charts'
ALLOWED_EXTENSIONS = {'csv'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create database tables
with app.app_context():
    db.create_all()
    logging.info("Database tables created successfully")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_structure(df):
    """Validate CSV has required columns"""
    required_columns = ['experience', 'rating']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for numeric columns
    try:
        pd.to_numeric(df['experience'], errors='raise')
        pd.to_numeric(df['rating'], errors='raise')
    except (ValueError, TypeError):
        return False, "Experience and rating columns must contain numeric values"
    
    return True, "Valid CSV structure"

def create_performance_labels(df):
    """Create performance labels based on rating"""
    def categorize_performance(rating):
        if rating >= 4.0:
            return 'High'
        elif rating >= 3.0:
            return 'Medium'
        else:
            return 'Low'
    
    df['performance'] = df['rating'].apply(categorize_performance)
    return df

def train_model(df, upload_id=None):
    """Train Random Forest model for performance prediction"""
    try:
        # Prepare features and target
        X = df[['experience', 'rating']]
        y = df['performance']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_path = os.path.join(MODELS_FOLDER, 'performance_model.pkl')
        joblib.dump(model, model_path)
        
        # Save model metrics to database if upload_id provided
        if upload_id:
            save_model_metrics_to_db(upload_id, y_test, y_pred, accuracy)
        
        logging.info(f"Model trained with accuracy: {accuracy:.4f}")
        return model, accuracy, y_test, y_pred
        
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def load_or_train_model(df, upload_id=None):
    """Load existing model or train new one"""
    model_path = os.path.join(MODELS_FOLDER, 'performance_model.pkl')
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logging.info("Existing model loaded successfully")
            return model, None, None, None
        except Exception as e:
            logging.warning(f"Failed to load existing model: {str(e)}")
    
    # Train new model if none exists or loading failed
    logging.info("Training new model...")
    return train_model(df, upload_id)

def generate_visualizations(df):
    """Generate data visualizations"""
    try:
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Employee Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Experience Distribution
        axes[0, 0].hist(df['experience'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Experience Distribution')
        axes[0, 0].set_xlabel('Years of Experience')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Rating Distribution
        axes[0, 1].hist(df['rating'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Rating Distribution')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Performance Categories
        performance_counts = df['performance'].value_counts()
        axes[1, 0].pie(performance_counts.values, labels=performance_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Performance Distribution')
        
        # 4. Experience vs Rating Scatter
        colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        for performance in df['performance'].unique():
            subset = df[df['performance'] == performance]
            axes[1, 1].scatter(subset['experience'], subset['rating'], 
                             label=performance, alpha=0.6, c=colors.get(performance, 'blue'))
        
        axes[1, 1].set_title('Experience vs Rating by Performance')
        axes[1, 1].set_xlabel('Years of Experience')
        axes[1, 1].set_ylabel('Rating')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(CHARTS_FOLDER, 'analysis_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
        return None

def make_predictions(model, df):
    """Make predictions using the trained model"""
    try:
        # Prepare features
        X = df[['experience', 'rating']]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['predicted_performance'] = predictions
        
        # Add confidence scores
        df_result['confidence'] = np.max(probabilities, axis=1)
        
        return df_result
        
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise

def save_upload_to_db(filename, original_filename, file_size, df):
    """Save upload information to database"""
    try:
        upload_record = UploadHistory(
            filename=filename,
            original_filename=original_filename,
            file_size=file_size,
            total_records=len(df),
            processing_status='pending'
        )
        db.session.add(upload_record)
        db.session.commit()
        return upload_record
    except Exception as e:
        logging.error(f"Error saving upload to database: {str(e)}")
        db.session.rollback()
        raise

def save_employee_data_to_db(upload_id, df):
    """Save employee data to database"""
    try:
        employee_records = []
        for index, row in df.iterrows():
            employee_record = EmployeeData(
                upload_id=upload_id,
                experience=row['experience'],
                rating=row['rating'],
                performance_category=row.get('performance'),
                predicted_performance=row.get('predicted_performance'),
                confidence_score=row.get('confidence')
            )
            employee_records.append(employee_record)
        
        db.session.add_all(employee_records)
        db.session.commit()
        logging.info(f"Saved {len(employee_records)} employee records to database")
        
    except Exception as e:
        logging.error(f"Error saving employee data to database: {str(e)}")
        db.session.rollback()
        raise

def save_model_metrics_to_db(upload_id, y_test, y_pred, accuracy):
    """Save model performance metrics to database"""
    try:
        # Calculate detailed metrics
        precision = precision_score(y_test, y_pred, average=None, labels=['High', 'Medium', 'Low'])
        recall = recall_score(y_test, y_pred, average=None, labels=['High', 'Medium', 'Low'])
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Map to High, Medium, Low order
        labels = ['High', 'Medium', 'Low']
        precision_dict = {label: precision[i] if i < len(precision) else 0.0 for i, label in enumerate(labels)}
        recall_dict = {label: recall[i] if i < len(recall) else 0.0 for i, label in enumerate(labels)}
        
        metrics_record = ModelMetrics(
            upload_id=upload_id,
            accuracy=accuracy,
            precision_high=precision_dict['High'],
            precision_medium=precision_dict['Medium'],
            precision_low=precision_dict['Low'],
            recall_high=recall_dict['High'],
            recall_medium=recall_dict['Medium'],
            recall_low=recall_dict['Low'],
            f1_score=f1
        )
        
        db.session.add(metrics_record)
        db.session.commit()
        logging.info(f"Saved model metrics to database")
        
    except Exception as e:
        logging.error(f"Error saving model metrics to database: {str(e)}")
        db.session.rollback()

def save_performance_stats_to_db(upload_id, df):
    """Save performance statistics to database"""
    try:
        stats_record = PerformanceStats(
            upload_id=upload_id,
            total_employees=len(df),
            high_performers=len(df[df['predicted_performance'] == 'High']),
            medium_performers=len(df[df['predicted_performance'] == 'Medium']),
            low_performers=len(df[df['predicted_performance'] == 'Low']),
            avg_experience=df['experience'].mean(),
            avg_rating=df['rating'].mean(),
            min_experience=df['experience'].min(),
            max_experience=df['experience'].max(),
            min_rating=df['rating'].min(),
            max_rating=df['rating'].max()
        )
        
        db.session.add(stats_record)
        db.session.commit()
        logging.info(f"Saved performance statistics to database")
        
    except Exception as e:
        logging.error(f"Error saving performance stats to database: {str(e)}")
        db.session.rollback()

def update_upload_status(upload_id, status, error_message=None, model_accuracy=None):
    """Update upload processing status"""
    try:
        upload_record = UploadHistory.query.get(upload_id)
        if upload_record:
            upload_record.processing_status = status
            if error_message:
                upload_record.error_message = error_message
            if model_accuracy:
                upload_record.model_accuracy = model_accuracy
            db.session.commit()
            
    except Exception as e:
        logging.error(f"Error updating upload status: {str(e)}")
        db.session.rollback()

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    upload_record = None
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file and get size
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        
        # Load and validate CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        # Validate CSV structure
        is_valid, message = validate_csv_structure(df)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('index'))
        
        # Save upload information to database
        upload_record = save_upload_to_db(filename, original_filename, file_size, df)
        
        # Create performance labels
        df = create_performance_labels(df)
        
        # Load or train model
        model, accuracy, y_test, y_pred = load_or_train_model(df, upload_record.id)
        
        # Make predictions
        df_result = make_predictions(model, df)
        
        # Save employee data to database
        save_employee_data_to_db(upload_record.id, df_result)
        
        # Save performance statistics to database
        save_performance_stats_to_db(upload_record.id, df_result)
        
        # Generate visualizations
        chart_path = generate_visualizations(df_result)
        
        # Save results
        result_filename = f"results_{timestamp}.csv"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        df_result.to_csv(result_filepath, index=False)
        
        # Update upload status to completed
        update_upload_status(upload_record.id, 'completed', model_accuracy=accuracy)
        
        # Prepare statistics
        stats = {
            'total_employees': len(df_result),
            'high_performers': len(df_result[df_result['predicted_performance'] == 'High']),
            'medium_performers': len(df_result[df_result['predicted_performance'] == 'Medium']),
            'low_performers': len(df_result[df_result['predicted_performance'] == 'Low']),
            'average_experience': df_result['experience'].mean(),
            'average_rating': df_result['rating'].mean(),
            'model_accuracy': accuracy
        }
        
        flash('File processed successfully!', 'success')
        
        return render_template('results.html', 
                             data=df_result.to_html(classes='table table-striped table-hover', 
                                                   table_id='results-table', escape=False),
                             stats=stats,
                             chart_available=chart_path is not None,
                             result_filename=result_filename,
                             upload_id=upload_record.id)
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        
        # Update upload status to failed if we have an upload record
        if upload_record:
            update_upload_status(upload_record.id, 'failed', error_message=str(e))
        
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed CSV file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            flash('File not found', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error downloading file: {str(e)}")
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/chart')
def view_chart():
    """Serve the analysis chart"""
    try:
        chart_path = os.path.join(CHARTS_FOLDER, 'analysis_chart.png')
        if os.path.exists(chart_path):
            return send_file(chart_path, mimetype='image/png')
        else:
            return "Chart not found", 404
    except Exception as e:
        logging.error(f"Error serving chart: {str(e)}")
        return "Error loading chart", 500

@app.route('/history')
def upload_history():
    """Display upload history"""
    try:
        uploads = UploadHistory.query.order_by(UploadHistory.upload_timestamp.desc()).limit(20).all()
        return render_template('history.html', uploads=uploads)
    except Exception as e:
        logging.error(f"Error fetching upload history: {str(e)}")
        flash(f'Error loading upload history: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/upload/<int:upload_id>')
def view_upload_details(upload_id):
    """View detailed information about a specific upload"""
    try:
        upload = UploadHistory.query.get_or_404(upload_id)
        employee_data = EmployeeData.query.filter_by(upload_id=upload_id).all()
        performance_stats = PerformanceStats.query.filter_by(upload_id=upload_id).first()
        model_metrics = ModelMetrics.query.filter_by(upload_id=upload_id).first()
        
        return render_template('upload_details.html', 
                             upload=upload,
                             employee_data=employee_data,
                             performance_stats=performance_stats,
                             model_metrics=model_metrics)
    except Exception as e:
        logging.error(f"Error fetching upload details: {str(e)}")
        flash(f'Error loading upload details: {str(e)}', 'error')
        return redirect(url_for('upload_history'))

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    try:
        total_uploads = UploadHistory.query.count()
        successful_uploads = UploadHistory.query.filter_by(processing_status='completed').count()
        failed_uploads = UploadHistory.query.filter_by(processing_status='failed').count()
        
        total_employees = db.session.query(db.func.sum(PerformanceStats.total_employees)).scalar() or 0
        
        # Get latest upload stats
        latest_upload = UploadHistory.query.order_by(UploadHistory.upload_timestamp.desc()).first()
        
        stats = {
            'total_uploads': total_uploads,
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'total_employees_analyzed': total_employees,
            'latest_upload': {
                'filename': latest_upload.original_filename if latest_upload else None,
                'timestamp': latest_upload.upload_timestamp.isoformat() if latest_upload else None,
                'status': latest_upload.processing_status if latest_upload else None
            } if latest_upload else None
        }
        
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error fetching API stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logging.error(f"Internal server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
