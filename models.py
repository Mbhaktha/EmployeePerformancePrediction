"""
Database models for HR Analytics application
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class UploadHistory(db.Model):
    """Track file uploads and processing history"""
    __tablename__ = 'upload_history'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)  # in bytes
    total_records = Column(Integer)
    processing_status = Column(String(50), default='pending')  # pending, completed, failed
    error_message = Column(Text)
    model_accuracy = Column(Float)
    
    def __repr__(self):
        return f'<UploadHistory {self.original_filename}>'

class EmployeeData(db.Model):
    """Store employee data from uploads"""
    __tablename__ = 'employee_data'
    
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, db.ForeignKey('upload_history.id'), nullable=False)
    experience = Column(Float, nullable=False)
    rating = Column(Float, nullable=False)
    performance_category = Column(String(20))  # High, Medium, Low
    predicted_performance = Column(String(20))
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    upload = db.relationship('UploadHistory', backref=db.backref('employee_records', lazy=True))
    
    def __repr__(self):
        return f'<EmployeeData {self.id}: {self.experience} years, {self.rating} rating>'

class ModelMetrics(db.Model):
    """Store model performance metrics"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, db.ForeignKey('upload_history.id'), nullable=False)
    accuracy = Column(Float)
    precision_high = Column(Float)
    precision_medium = Column(Float)
    precision_low = Column(Float)
    recall_high = Column(Float)
    recall_medium = Column(Float)
    recall_low = Column(Float)
    f1_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    upload = db.relationship('UploadHistory', backref=db.backref('model_metrics', lazy=True))
    
    def __repr__(self):
        return f'<ModelMetrics {self.id}: Accuracy {self.accuracy:.4f}>'

class PerformanceStats(db.Model):
    """Store aggregated performance statistics"""
    __tablename__ = 'performance_stats'
    
    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, db.ForeignKey('upload_history.id'), nullable=False)
    total_employees = Column(Integer)
    high_performers = Column(Integer)
    medium_performers = Column(Integer)
    low_performers = Column(Integer)
    avg_experience = Column(Float)
    avg_rating = Column(Float)
    min_experience = Column(Float)
    max_experience = Column(Float)
    min_rating = Column(Float)
    max_rating = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    upload = db.relationship('UploadHistory', backref=db.backref('performance_stats', lazy=True))
    
    def __repr__(self):
        return f'<PerformanceStats {self.id}: {self.total_employees} employees>'

class AppSettings(db.Model):
    """Store application settings and configurations"""
    __tablename__ = 'app_settings'
    
    id = Column(Integer, primary_key=True)
    setting_key = Column(String(100), unique=True, nullable=False)
    setting_value = Column(Text)
    setting_type = Column(String(20), default='string')  # string, integer, float, boolean
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<AppSettings {self.setting_key}: {self.setting_value}>'