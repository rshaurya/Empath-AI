# Empath AI
A **multimodal** AI chatbot for emotion detection. EmpathAI is an AI-powered chatbot designed to detect and interpret human emotions through audio and textual inputs. By combining natural language processing (NLP) andspeech signal analysis, the system aims to provide emotionally intelligent interactions.

---

## About the project
The project uses:
- Speech analysis to detect pitch, prosody, and emotion.
- Text processing to understand sentiment, intent, and ambiguity.
- Future extensions include facial expression analysis via video input, enabling full
multimodal emotion detection.
- This project is a **FastAPI microservice** for text sentiment analysis using **classic machine learning** (TF-IDF + Logistic Regression + SVM).  
- It can be trained on datasets like [GoEmotions](https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/).  

---

## 🚀 Current Features (as per the latest update)  
- Train your own sentiment classifier from a CSV dataset (`text,label` format).
- Predict fine-grained emotions like *joyful, angry, hopeful, disappointed*.
- Predict top 'k' emotions corresponding to a particular text.
- Lightweight and resource-friendly (no heavy LLMs required).
- Deployable as a **Docker container**.

---

## File structure (think of another name) 

## Project Setup

### 1. Clone Repository
```bash
git clone https://github.com/rshaurya/Empath-AI.git
cd Empath-AI
```

### 2. Create a Virtual Environment
Make sure you have **Python 3.11** installed. To verify installation:

```bash
python3.11 --version
```
If a version appears, then **Python3.11** is installed.

Create a virtual environment: 
```bash
python3.11 -m venv .venv
```

Activate the environment(Linux/Mac)
```bash
source .venv/bin/activate
```

Activate the environement (Windows PowerShell)
```bash
.venv\Scripts\activate
```

### 3. Install requirements
Navigate to the `Empath-AI` directory and run the following command to install the requirements:
```bash
pip install -r requirements.txt
```


By now, you must have created a virutal environment, activated it and installed all the dependencies. Follow further steps to run the application.

---
## Running the Application

### 1. Merge CSV files
There are 3 files under `train/`. Run the following command to merge the 3 `.csv` files into one:
```bash
cat train/goemotions_*.csv > train/goemotions_full.csv
```
This will create a new file inside `train/` 

### 2. Data Preprocessing
The `train/goemotions_full.csv` has a lot of columns like `text, id, author_id, date` etc. out of which only `text` and `label` are necessary for training our model. 

- Run the `bin/re-format-goemotions.py` file using any code editor or run the following command:
```bash
python3.11 bin/re-format-goemotions.py
```
This will create a new file, `train/goemotions_text_label.csv` having only 2 columns i.e. `text` and `label`.



### 3. Run server
To run and test the application run the following command from the project root:

```bash
uvicorn app.main:app --reload  --port 8000
```

Now open your browser at: [http://localhost:8000/docs](http://localhost:8000/docs)  
This gives you an **interactive Swagger UI** where you can try training and prediction endpoints. 
This is where you can test your API endpoints, just click on a particular end point and click the "Try it out" button. **Swagger** tells you a lot about your API. It tells you:

1. The type of method (GET, POST etc.).
1. The parameters required (if any) with its type.
1. the response you can expect from the API.
1. And the response type in detail.

### 4. Train the Model
- In the Swagger UI, select the `/train` endpoint (POST) and click on `Try it out`.
- Keep the default parameter `use_logreg` as `false`.
- Under the `request_body` upload the `goemotions_text_label.csv` file from your system.
- Then click execute. And let it do its magic. The model is being trained!
- This step will create the model `model.joblib` in the `app/` directory.

### 5. Predict text
- Now, select the `predict/` endpoint (POST) and click on `Try it out`.
- Under the `request_body` edit the values in json:
```json
{
  "text": "string",
  "top_k": 1
}
```
- Here, `text` contains the sentence for which we have to detect the emotion. And `top_k` means the top 1 or 2 or 3 emotions corresponding to the particular text. It controls how many top labels are returned 


Response:
```json
{
  "text": "I am so happy today!",
  "prediction": "joyful"
}
```

---

Yay! The text sentiment analysis is completed! More features coming soon...
