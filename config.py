import os

class Config:
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production-2025'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = 'sqlite:///medical_system.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session configuration
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Admin credentials (for default admin account)
    ADMIN_USERNAME = 'admin'
    ADMIN_EMAIL = 'admin@medicalsystem.com'
    ADMIN_PASSWORD = 'admin123'  
