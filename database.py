from app import app, db

def initialize_database():
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")
        print("✅ You can now register users and save calculations!")

if __name__ == "__main__":
    initialize_database()