# Orion - Scotiabank/Tangerine

This project provides a Flask-based API to detect potential credit card fraud on websites. It checks whether a website is whitelisted, blacklisted, or potentially fraudulent by analyzing its HTML content, identifying fields that might be requesting credit card information, and interacting with suspicious endpoints.

## Features

- **Whitelist/Blacklist Checking:** The API checks if a website is in the whitelist or blacklist before proceeding with further analysis.
- **HTML Parsing:** Uses BeautifulSoup to parse HTML content and identify fields that might be requesting sensitive credit card information.
- **Credit Card Fraud Detection:** Simulates the submission of false credit card information to detect fraudulent activity.
- **Confidence Scoring:** Utilizes a PyTorch model to generate a confidence score for the likelihood of fraud.

## Setup

### Create a Virtual Environment

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

### Install Dependencies

- **Install the required packages:**
  ```bash
  pip install -r requirements.txt
  ```

### Run Your App Locally in Modal Cloud

- **Run your app:**
  ```bash
  modal serve src.entrypoint
  ```

### Deploy Your App

- **Deploy your application to Modal cloud and get a URL:**
  ```bash
  modal deploy src.entrypoint --name <name of app>
  ```

## API Endpoints

### 1. **Home Endpoint**
   - **URL:** `/`
   - **Method:** GET
   - **Description:** Returns a welcome message.
   - **Response:**
     ```json
     {
       "message": "Welcome to the Credit Card Checker API!"
     }
     ```

### 2. **Test Endpoint**
   - **URL:** `/test`
   - **Method:** POST
   - **Description:** Simulates a check on a whitelisted website.
   - **Response:**
     ```json
     {
       "reason": "Whitelisted website.",
       "secure": True,
       "confidence_score": 100
     }
     ```

### 3. **Check Credit Card Endpoint**
   - **URL:** `/check`
   - **Method:** POST
   - **Description:** Analyzes a website's HTML content for potential credit card fraud.
   - **Request Body:**
     ```json
     {
       "url": "https://example.com",
       "content": "<html>...</html>"
     }
     ```
   - **Response:**
     - If the website is whitelisted:
       ```json
       {
         "reason": "Whitelisted website.",
         "secure": True,
         "confidence_score": 100
       }
       ```
     - If the website is blacklisted:
       ```json
       {
         "reason": "Blacklisted website.",
         "secure": False,
         "confidence_score": 0
       }
       ```
     - If potential fraud is detected:
       ```json
       {
         "reason": "The website is doing fraud.",
         "secure": False,
         "confidence_score": 100
       }
       ```
     - If no fraud is detected:
       ```json
       {
         "message": "No credit card information requested.",
         "found": False
       }
       ```

## Whitelist and Blacklist

The application uses predefined lists of whitelisted and blacklisted websites:

- **Whitelisted Websites:** 
  - `https://www.amazon.com`
  - `https://www.amazon.co.uk`
  - `https://www.amazon.ca`
  - `https://www.amazon.com.au`
  - `https://www.amazon.in`

- **Blacklisted Websites:**
  - `https://temu.bappenas.go.id`
  - `https://www.temu.com/`
  - `https://www.arshiazr.vercel.app`

The API also checks for blacklisted API endpoints that might be used for fraudulent activity.

## License

This project is licensed under the MIT License.
