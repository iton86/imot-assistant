import os
from flask import Flask, render_template, jsonify, request
from collections import defaultdict
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, TIMESTAMP, func, and_, or_
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
    ad_price_per_kvm = db.Column(db.Numeric(precision=10, scale=2))
    ad_kvm = db.Column(db.Integer)
    ad_street = db.Column(db.String)
    locations = db.Column(db.String)
    updated_ts = db.Column(TIMESTAMP, nullable=False)
    ad_show = db.Column(db.Boolean, default=True)

# Route to render the table and handle user actions
@app.route('/')
def index():
    data = YourModel.query.all()
    distinct_values = [value[0] for value in db.session.query(YourModel.ad_type).distinct().all()]
    distinct_values = sorted(distinct_values)

    for item in data:
        item.alias = item.ad_url.split('=')[-1]

    return render_template('dashboard.html', data=data, distinct_values=distinct_values)

# API endpoint to fetch updated data for the table
@app.route('/get_data', methods=['GET'])
def get_data():
    data = YourModel.query.filter_by(ad_show=True).all()
    result = [{
               'pic_url': item.pic_url,
               'alias': item.alias,
               'ad_url': item.ad_url,
               'ad_type': item.ad_type,
               'ad_price': item.ad_price,
               'ad_per_kvm': item.price_per_kvm,
               'ad_kvm': item.kvm,
               'ad_street': item.ad_street,
               'locations': item.locations,
               'updated_ts': item.updated_ts} for item in data]

    return jsonify(result)

# API endpoint to handle 'remove' action and update the database
@app.route('/update_data', methods=['POST'])
def update_data():
    id_to_remove = request.json['id']
    row_to_remove = YourModel.query.get(id_to_remove)
    row_to_remove.ad_show = False
    db.session.commit()
    return jsonify({'message': 'Row updated successfully'})

@app.route('/filter_by_values', methods=['POST'])
def filter_by_values():
    selected_values = request.json['selectedValues']
    filtered_data = YourModel.query.filter(and_(YourModel.ad_type.in_(selected_values),
                                                YourModel.ad_show, YourModel.ad_price > 0)).all()

    group_averages = defaultdict(list)

    for row in filtered_data:
        group, value = row.ad_type, row.ad_price_per_kvm
        group_averages[group].append(value)

    average_by_group = {}
    for group, values in group_averages.items():
        average_by_group[group] = round(sum(values) / len(values), 2) if values else 0

    for item in filtered_data:
        item.gain = round((item.ad_price_per_kvm / average_by_group[item.ad_type] - 1) * 100, 0)

    filtered_data = sorted(filtered_data, key=lambda x: x.gain)

    serialized_data = [{
               'pic_url': item.pic_url,
               'gain': item.gain,
               'ad_url': item.ad_url,
               'ad_type': item.ad_type,
               'ad_price': item.ad_price,
               'ad_price_per_kvm': item.ad_price_per_kvm,
               'ad_kvm': item.ad_kvm,
               'ad_street': item.ad_street,
               'locations': item.locations,
               'updated_ts': item.updated_ts.strftime('%Y-%m-%d %H:%M:%S')} for item in filtered_data]

    return jsonify({'filteredData': serialized_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
