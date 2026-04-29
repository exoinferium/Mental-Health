from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
import tensorflow as tf
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import string
import json
import os

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_SORT_KEYS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# ============ DATABASE MODELS ============

class MoodEntry(db.Model):
    __tablename__ = 'mood_entries'
    id = db.Column(db.Integer, primary_key=True)
    emoji = db.Column(db.String(10), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    note = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    date = db.Column(db.String(50))

    def to_dict(self):
        return {
            'id': self.id,
            'emoji': self.emoji,
            'label': self.label,
            'note': self.note,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'date': self.date
        }

class JournalEntry(db.Model):
    __tablename__ = 'journal_entries'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    date = db.Column(db.String(50))

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'date': self.date
        }

class ChatbotResponse(db.Model):
    __tablename__ = 'chatbot_responses'
    id = db.Column(db.Integer, primary_key=True)
    user_query = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_query': self.user_query,
            'bot_response': self.bot_response,
            'category': self.category,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

class FeedbackForm(db.Model):
    __tablename__ = 'feedback_forms'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer)  # 1-5 rating
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'message': self.message,
            'rating': self.rating,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

class AppFeelingsForm(db.Model):
    __tablename__ = 'app_feelings_forms'
    id = db.Column(db.Integer, primary_key=True)
    helpful = db.Column(db.String(10))  # yes/no
    favorite_feature = db.Column(db.String(100))
    improve = db.Column(db.Text)
    continue_using = db.Column(db.String(10))  # yes/no
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'helpful': self.helpful,
            'favorite_feature': self.favorite_feature,
            'improve': self.improve,
            'continue_using': self.continue_using,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

# ============ CHATBOT SETUP ============

stemmer = LancasterStemmer()

# Download nltk dependencies
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# Load and prepare data
categories = []
questions = []
answers = []

# Load training data
try:
    with open('backend/aichatbot.txt', 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            categories.append(line)
            questions.append(f.readline().lower().strip())
            answers.append(f.readline().lower().strip())
except FileNotFoundError:
    print("Warning: aichatbot.txt not found. Chatbot will not be available.")
    categories = ['default']
    questions = ['hello']
    answers = ['hi there!']

# Tokenize and remove stop words
word_tokens_stop = []
questions_tokenized_stopped = []
for i, question in enumerate(questions):
    question = question.translate(punct_dict)
    tokens = nltk.word_tokenize(question)
    tokens_stop = [w for w in tokens if not w in stop_words]
    word_tokens_stop.extend(tokens_stop)
    questions_tokenized_stopped.append(tokens_stop)

# Stem words
stemmed_words = [stemmer.stem(w) for w in word_tokens_stop]
stemmed_words = sorted(list(set(stemmed_words)))

sorted_categories = sorted(list(set(categories)))

# Prepare training data
training = []
output = []

for i, question in enumerate(questions_tokenized_stopped):
    training_row = []
    stemmed_question = [stemmer.stem(token) for token in question]

    for w in stemmed_words:
        training_row.append(1 if w in stemmed_question else 0)

    output_row = [0] * len(sorted_categories)
    output_row[sorted_categories.index(categories[i])] = 1

    training.append(training_row)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Build the neural network
input_size = len(training[0])
output_size = len(output[0])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_size,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training, output, epochs=1000, batch_size=8, verbose=0)

# Function to process input and predict category
def get_response(query):
    row = [0] * len(stemmed_words)
    query = query.lower().translate(punct_dict)
    tokens = nltk.word_tokenize(query)
    tokens_stop = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens_stop]

    for stemmed_word in stemmed_tokens:
        for i, w in enumerate(stemmed_words):
            if w == stemmed_word:
                row[i] = 1

    return np.array(row)

# ============ ROUTES ============

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

# ============ CHATBOT ROUTES ============

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({'error': 'Query is required'}), 400

        response_vector = get_response(user_query)
        results = model.predict(np.array([response_vector]), verbose=0)
        results_index = np.argmax(results)
        tag = sorted_categories[results_index]

        # Get random response from category
        responses = [answers[i] for i, category in enumerate(categories) if category == tag]
        bot_response = random.choice(responses)

        # Save to database
        chat_entry = ChatbotResponse(
            user_query=user_query,
            bot_response=bot_response,
            category=tag
        )
        db.session.add(chat_entry)
        db.session.commit()

        return jsonify({
            'response': bot_response,
            'category': tag,
            'confidence': float(results[0][results_index])
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
def chat_history():
    try:
        history = ChatbotResponse.query.order_by(ChatbotResponse.timestamp.desc()).limit(50).all()
        return jsonify([h.to_dict() for h in history]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ MOOD TRACKER ROUTES ============

@app.route('/api/mood', methods=['POST'])
def add_mood():
    try:
        data = request.json
        mood_entry = MoodEntry(
            emoji=data.get('emoji'),
            label=data.get('label'),
            note=data.get('note', ''),
            date=data.get('date')
        )
        db.session.add(mood_entry)
        db.session.commit()
        return jsonify(mood_entry.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mood', methods=['GET'])
def get_moods():
    try:
        moods = MoodEntry.query.order_by(MoodEntry.timestamp.desc()).all()
        return jsonify([m.to_dict() for m in moods]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ JOURNAL ROUTES ============

@app.route('/api/journal', methods=['POST'])
def add_journal():
    try:
        data = request.json
        journal_entry = JournalEntry(
            title=data.get('title'),
            content=data.get('content'),
            date=data.get('date')
        )
        db.session.add(journal_entry)
        db.session.commit()
        return jsonify(journal_entry.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/journal', methods=['GET'])
def get_journal():
    try:
        entries = JournalEntry.query.order_by(JournalEntry.timestamp.desc()).all()
        return jsonify([j.to_dict() for j in entries]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/journal/<int:entry_id>', methods=['DELETE'])
def delete_journal(entry_id):
    try:
        entry = JournalEntry.query.get(entry_id)
        if not entry:
            return jsonify({'error': 'Entry not found'}), 404
        db.session.delete(entry)
        db.session.commit()
        return jsonify({'message': 'Entry deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ FEEDBACK FORM ROUTES ============

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        feedback = FeedbackForm(
            name=data.get('name'),
            email=data.get('email'),
            message=data.get('message'),
            rating=data.get('rating')
        )
        db.session.add(feedback)
        db.session.commit()
        return jsonify(feedback.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    try:
        feedbacks = FeedbackForm.query.order_by(FeedbackForm.timestamp.desc()).all()
        return jsonify([f.to_dict() for f in feedbacks]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ APP FEELINGS FORM ROUTES ============

@app.route('/api/app-feelings', methods=['POST'])
def submit_app_feelings():
    try:
        data = request.json
        form = AppFeelingsForm(
            helpful=data.get('helpful'),
            favorite_feature=data.get('favorite_feature'),
            improve=data.get('improve'),
            continue_using=data.get('continue_using')
        )
        db.session.add(form)
        db.session.commit()
        return jsonify(form.to_dict()), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/app-feelings', methods=['GET'])
def get_app_feelings():
    try:
        forms = AppFeelingsForm.query.order_by(AppFeelingsForm.timestamp.desc()).all()
        return jsonify([f.to_dict() for f in forms]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ STATISTICS ROUTES ============

@app.route('/api/stats/mood-distribution', methods=['GET'])
def mood_distribution():
    try:
        moods = MoodEntry.query.all()
        distribution = {}
        for mood in moods:
            distribution[mood.label] = distribution.get(mood.label, 0) + 1
        return jsonify(distribution), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/dashboard', methods=['GET'])
def dashboard_stats():
    try:
        total_moods = MoodEntry.query.count()
        total_journals = JournalEntry.query.count()
        total_chats = ChatbotResponse.query.count()
        total_feedback = FeedbackForm.query.count()

        return jsonify({
            'total_mood_entries': total_moods,
            'total_journal_entries': total_journals,
            'total_chat_messages': total_chats,
            'total_feedback_forms': total_feedback
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ DATABASE INITIALIZATION ============

@app.route('/api/init-db', methods=['POST'])
def init_db():
    try:
        db.create_all()
        return jsonify({'message': 'Database initialized'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
