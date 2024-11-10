from flask import Flask, request, jsonify, render_template, session
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
import os
import re
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv

load_dotenv()


# Ensure NLTK stopwords data is downloaded
nltk.download('stopwords')

class ResearchPaperPreprocessor:
    def __init__(self, filepath, chunk_size=1000):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.raw_text = ""
        self.abstract = ""
        self.body_text = ""
        self.references = []
        self.processed_data = {'abstract': '', 'body_chunks': [], 'references': []}

    def load_text(self):
        # Load the text from file
        with open(self.filepath, 'r', encoding='utf-8') as file:
            self.raw_text = file.read()
    
    def clean_text(self, text):
        # Lowercase, remove unwanted characters, and extra whitespace
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text):
        # Remove English stop words
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_text = ' '.join(word for word in words if word not in stop_words)
        return filtered_text
    
    def extract_sections(self):
        # Extract abstract and body sections based on keywords
        abstract_match = re.search(r'abstract(.*?)introduction', self.raw_text, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            self.abstract = abstract_match.group(1)
        else:
            self.abstract = self.raw_text[:1000]  # Default if abstract not found

        body_match = re.search(r'introduction(.*?)(references|$)', self.raw_text, re.DOTALL | re.IGNORECASE)
        if body_match:
            self.body_text = body_match.group(1)
        else:
            self.body_text = self.raw_text

        # Extract references section
        references_match = re.search(r'references(.*)', self.raw_text, re.DOTALL | re.IGNORECASE)
        if references_match:
            references_text = references_match.group(1)
            # Split references based on line breaks or numbering
            self.references = [ref.strip() for ref in re.split(r'\n|\d+\.\s', references_text) if ref.strip()]

    def preprocess_text(self):
        # Clean and remove stop words from abstract and body
        self.abstract = self.clean_text(self.abstract)
        self.abstract = self.remove_stopwords(self.abstract)
        self.body_text = self.clean_text(self.body_text)
        self.body_text = self.remove_stopwords(self.body_text)

    def split_into_chunks(self):
        # Split body text into chunks based on specified chunk size
        # Using a simple split by common sentence-ending punctuation.
        sentences = re.split(r'(?<=[.!?]) +', self.body_text)
        chunk = ""
        
        for sentence in sentences:
            if len(chunk) + len(sentence) < self.chunk_size:
                chunk += " " + sentence
            else:
                self.processed_data['body_chunks'].append(chunk.strip())
                chunk = sentence

        if chunk:  # Add any remaining text as the last chunk
            self.processed_data['body_chunks'].append(chunk.strip())

    def process(self):
        # Run the full preprocessing pipeline
        self.load_text()
        self.extract_sections()
        self.preprocess_text()
        self.split_into_chunks()
        
        # Store processed data
        self.processed_data['abstract'] = self.abstract
        self.processed_data['references'] = self.references
        return self.processed_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
UPLOAD_FOLDER = 'uploads'  # Folder to save uploaded files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
huggingface_api_key = os.getenv('HUGGINGFACE_KEY')
qdrant_api_key = os.getenv('QDRANT_KEY')
qdrant_url = os.getenv('QDRANT_URL')

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=huggingface_api_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)
url = qdrant_url
api_key = qdrant_api_key

client = QdrantClient(url=url, api_key=api_key)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

def longest_common_substring(text1, text2):
    # Normalize the texts by splitting into words
    words1 = text1.split()
    words2 = text2.split()

    # Create a set to store common substrings
    common_substrings = set()

    # Generate substrings from the first text
    for i in range(len(words1)):
        for j in range(i + 1, len(words1) + 1):
            substring = ' '.join(words1[i:j])  # Create substring
            if substring in text2:  # Check if substring exists in text2
                common_substrings.add(substring)

    # Find the longest common substring
    longest_substring = ''
    for substring in common_substrings:
        if len(substring) > len(longest_substring):
            longest_substring = substring

    # Return the number of words in the longest common substring
    return len(longest_substring.split()), longest_substring

def remove_txt_extension(filename):
    return filename.replace('.txt', '') if filename.endswith('.txt') else filename

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading text
@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    if 'text' in data:
        text_content = data['text']
        
        # Save the text to a file with utf-8 encoding
        file_name = 'uploaded_text.txt'  # You can make this unique if desired
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as file:  # Use utf-8 encoding
                file.write(text_content)
            
            # Store the file path in session
            session['file_path'] = file_path
            print(f"Saved text to: {file_path}")  # For demonstration, print to console
            
            return jsonify({'message': 'Text uploaded successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Handle exceptions when writing the file

    return jsonify({'error': 'No text provided'}), 400

vector_store = QdrantVectorStore(
    client=client,
    collection_name="plagiarism_collection3",
    embedding=embeddings,
)

# Route for displaying the result
@app.route('/result')
def result():
    return render_template('result.html')
    # Retrieve the file path from the session

@app.route('/check', methods=['GET'])
def check():
    file_path = session.get('file_path')
    preprocessor = ResearchPaperPreprocessor(filepath=file_path, chunk_size=10000)
    processed_data = preprocessor.process()
    content =  processed_data['abstract'] + "".join(processed_data['body_chunks'])
    referencelist = []
    for ref in processed_data['references']:
        referencelist.append(remove_txt_extension(ref))
    print(referencelist)
    texts = text_splitter.create_documents([content])
    total = 0
    plags = []
    for text in texts:
        try: 
            result = vector_store.similarity_search_with_score(text.page_content, k=1)
            score = result[0][1]
            src = remove_txt_extension(result[0][0].metadata['source'])
            page = result[0][0].page_content
            if score > 0.8 and src not in referencelist:
                count, plag = longest_common_substring(text.page_content, page)
                total += count
                plags.append(plag)
        except:
            pass
    percent = total/len(content.split())*100
    percent = round(percent, 2)
    return jsonify({'plagiarism_score': percent, 'content': content, 'plags': plags})

if __name__ == '__main__':
    app.run(debug=True)