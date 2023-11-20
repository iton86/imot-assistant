import os
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

db_user = os.getenv('POSTGRES_USER')
db_pass = os.getenv('POSTGRES_PASSWORD')
db_host = os.getenv('POSTGRES_HOST_IMOT')  # _DOCKER for local run, _IMOT for DOCKER-COMPOSE
db_port = os.getenv('POSTGRES_PORT')
db_database = os.getenv('POSTGRES_DB')

# Configure your PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_database}'
# app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://postgres:admin@localhost:5432/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define your SQLAlchemy model
class YourModel(db.Model):
    __tablename__ = 'ads_latest'

    pic_url = db.Column(db.String)
    ad_url = db.Column(db.String, primary_key=True)
    ad_type = db.Column(db.String)
    ad_price = db.Column(db.Integer)
    locations = db.Column(db.String)
    ad_show = db.Column(db.Boolean, default=True)

# Route to render the table and handle user actions
@app.route('/')
def index():
    data = YourModel.query.all()
    for item in data:
        item.alias = item.ad_url.split('=')[-1]

    return render_template('dashboard.html', data=data)

# API endpoint to fetch updated data for the table
@app.route('/get_data', methods=['GET'])
def get_data():
    data = YourModel.query.filter_by(ad_show=True).all()
    result = [{
               'pic_url': item.pic_url,
               'ad_url': item.ad_url,
               'ad_type': item.ad_type,
               'ad_price': item.ad_price,
               'locations': item.locations} for item in data]
    return jsonify(result)

# API endpoint to handle 'remove' action and update the database
@app.route('/update_data', methods=['POST'])
def update_data():
    id_to_remove = request.json['id']
    row_to_remove = YourModel.query.get(id_to_remove)
    row_to_remove.ad_show = False
    db.session.commit()
    return jsonify({'message': 'Row updated successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
