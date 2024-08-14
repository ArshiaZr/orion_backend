import modal
from .FraudModel import tokenize_html, FraudDetectionModel

app = modal.App()

image = modal.Image.debian_slim().pip_install(
    "flask",
    "flask_cors",
    "requests",
    "beautifulsoup4",
    "torch",
"numpy",
"transformers"
)


whitelist = [
    "https://www.amazon.com",
    "https://www.amazon.co.uk",
    "https://www.amazon.ca",
    "https://www.amazon.com.au",
    "https://www.amazon.in",
]

blacklist = [
    'https://temu.bappenas.go.id',
    "https://www.temu.com/",
    "https://www.arshiazr.vercel.app"
]

API_blacklist = [
    'https://temu.bappenas.go.id',
]

check_list = ['cc-number', 'credit-card', 'cardnumber', 'card-number', 'ccn', 'ccnumber', 'cc_num', 'card_number', 'cardNumber']


@app.function(image=image)
@modal.wsgi_app()
def flask_app():
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    import json
    import requests
    from bs4 import BeautifulSoup
    import torch

    input_dim = 768
    hidden_dim = 64  # You can adjust the hidden layer size

    model = FraudDetectionModel(input_dim, hidden_dim)

    web_app = Flask(__name__)
    CORS(web_app)


    @web_app.get("/")
    def home():
        return jsonify({"message": "Welcome to the Credit Card Checker API!"})
    
    @web_app.post("/test")
    def test():
         return jsonify({"reason": "Whitelisted website.", "secure": True, "confidence_score": 100}), 200
    
    @web_app.post("/check")
    def check_credit_card():
        # Get the URL from the request
        data = json.loads(request.data)
        url = data['url']
        html_content = data['content']

        # Check whitelist
        if check_whitelisted(url):
            return jsonify({"reason": "Whitelisted website.", "secure": True, "confidence_score": 100}), 200
        
        # Check blacklist
        if check_blacklisted(url):
            return jsonify({"reason": "Blacklisted website.", "secure": False, "confidence_score": 0}), 200
        
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')


        # Look for input fields that might be requesting credit card info
        cc_fields = soup.find_all('input', {'name': check_list})
        if len(cc_fields) == 0:
            cc_fields = soup.find_all('input', {'autocomplete': check_list})

        if len(cc_fields) == 0: 
            # check if there's an iframe in the page that has the credit
            cc_fields = soup.find_all('iframe', {'name': check_list})

        if len(cc_fields) == 0:
            cc_fields = soup.find_all(['div', 'input', 'select', 'textarea'], {'name': check_list})

        if len(cc_fields) == 0:
            cc_fields = soup.find_all(['div', 'input', 'select', 'textarea'], {'id': check_list})

        # see documentation for the iframe tag
        if len(cc_fields) == 0:
            cc_fields = soup.find_all('iframe', {'autocomplete': check_list})
        
        if len(cc_fields) == 0:
            return jsonify({"message": "No credit card information requested.", "found": False}), 200
        
        # Sends false credit card info to the server
        credit_card_info = {
            "cc-number": "1234567812345678",
            "cc-exp": "12/27",
            "cc-cvv": "123"
        }

        # find the submit button and retrive the api endpoint
        form = cc_fields[0].find_parent('form')
        if not form:
            return jsonify({"reason": "Invalid API call", "secure": False, "confidence_score": 100}), 200
        submit_button = form.find('button', type='submit')
        if not submit_button:
            return jsonify({"reason": "Invalid API call", "secure": False, "confidence_score": 100}), 200
       
        endpoint = None
        method = None
        # check if the form has an action and method
        if form.has_attr('action'):
            endpoint = form['action']
        if form.has_attr('method'):
            method = form['method']
        
        if not endpoint:    
            endpoint = url           

        if not method:
            method = 'POST' 

        # Check if the API endpoint is blacklisted
        if check_blacklistAPIs(endpoint):
            return jsonify({"reason": "Blacklisted API endpoint.", "secure": False, "confidence_score": 100}), 200
        
        # Send the false credit card info to the server
        response = requests.request(method, endpoint, data=credit_card_info)
        if response.status_code == 200:
            return jsonify({"reason": "The website is doing fraude", "secure": False, "confidence_score": 100}), 200
                
        # generate a fake embedding
        embeddings = torch.randn(768)
        confidence_score = model(embeddings)
        confidence_score = confidence_score.item() * 100
        return jsonify({"reason": "Proceed it with your own risk", "secure": None, "confidence_score": confidence_score}), 200

        
    return web_app

def check_whitelisted(url):
    if url in whitelist:
        return True
    return False

def check_blacklisted(url):
    if url in blacklist:
        return True
    return False

def check_blacklistAPIs(url):
    if url in API_blacklist:
        return True
    return False
