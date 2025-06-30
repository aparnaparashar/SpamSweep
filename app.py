from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from urllib.parse import urlparse
from tld import get_tld
import re

app = Flask(__name__)

# Load models and vectorizers
email_model = pickle.load(open("email_model.pkl", "rb"))
email_vectorizer = pickle.load(open("email_vectorizer.pkl", "rb"))

sms_model = pickle.load(open("model.pkl", "rb"))
sms_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

url_model = pickle.load(open("xgb_c.pkl", "rb"))

# --------------------- URL Feature Engineering ---------------------

def having_ip_address(url):
    match = re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])', url)
    return 1 if match else 0

def abnormal_url(url):
    hostname = urlparse(url).hostname
    return 1 if hostname and hostname in url else 0

def count_dot(url): return url.count('.')
def count_www(url): return url.count('www')
def count_atrate(url): return url.count('@')
def no_of_dir(url): return urlparse(url).path.count('/')
def no_of_embed(url): return urlparse(url).path.count('//')

def shortening_service(url):
    match = re.search(r'bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|buff\.ly|adf\.ly', url)
    return 1 if match else 0

def count_https(url): return url.count('https')
def count_http(url): return url.count('http')

def fd_length(url):
    path = urlparse(url).path
    try:
        return len(path.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def extract_url_features(url):
    features = [
        having_ip_address(url),
        abnormal_url(url),
        count_dot(url),
        count_www(url),
        count_atrate(url),
        no_of_dir(url),
        no_of_embed(url),
        shortening_service(url),
        count_https(url),
        count_http(url),
        fd_length(url),
        tld_length(get_tld(url, fail_silently=True))
    ]
    return np.array(features).reshape(1, -1)

# --------------------- Routes ---------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    scan_type = data.get('type')
    content = data.get('content')

    if not scan_type or not content:
        return jsonify({'result': 'Invalid request'}), 400

    if scan_type == 'email':
        vec = email_vectorizer.transform([content])
        pred = email_model.predict(vec)[0]
        result = 'Spam' if pred == 1 else 'Ham'

    elif scan_type == 'sms':
        vec = sms_vectorizer.transform([content])
        pred = sms_model.predict(vec)[0]
        result = 'Spam' if pred == 1 else 'Ham'

    elif scan_type == 'url':
        features = extract_url_features(content)
        pred = url_model.predict(features)[0]
        label_map = {0: 'SAFE', 1: 'DEFACEMENT', 2: 'PHISHING', 3: 'MALWARE'}
        result = label_map.get(int(pred), 'Unknown')

    else:
        return jsonify({'result': 'Invalid scan type'}), 400

    return jsonify({'result': result})

# --------------------- Run App ---------------------

if __name__ == '__main__':
    app.run(debug=True)
