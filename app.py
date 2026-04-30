from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import string
import json
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "mindcare.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ============ DATABASE MODELS ============

class MoodEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emoji = db.Column(db.String(10))
    label = db.Column(db.String(50))
    note = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'emoji': self.emoji,
            'label': self.label,
            'note': self.note,
            'timestamp': self.timestamp.isoformat()
        }

class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

class ChatbotResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.Text)
    bot_response = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_message': self.user_message,
            'bot_response': self.bot_response,
            'timestamp': self.timestamp.isoformat()
        }

class FeedbackForm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    email = db.Column(db.String(200))
    feedback = db.Column(db.Text)
    rating = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'feedback': self.feedback,
            'rating': self.rating,
            'timestamp': self.timestamp.isoformat()
        }

class AppFeelingsForm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_name = db.Column(db.String(200))
    features_used = db.Column(db.String(500))
    feeling_about_app = db.Column(db.String(50))
    improvement_suggestions = db.Column(db.Text)
    would_recommend = db.Column(db.Boolean)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'app_name': self.app_name,
            'features_used': self.features_used,
            'feeling_about_app': self.feeling_about_app,
            'improvement_suggestions': self.improvement_suggestions,
            'would_recommend': self.would_recommend,
            'timestamp': self.timestamp.isoformat()
        }

# ============ CHATBOT INITIALIZATION ============

stemmer = LancasterStemmer()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# Load and prepare chatbot data
categories = []
questions = []
answers = []

# Load training data from aichatbot.txt
try:
    with open("aichatbot.txt", "r") as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            categories.append(line)
            questions.append(f.readline().lower().strip())
            answers.append(f.readline().lower().strip())
except FileNotFoundError:
    print("Warning: aichatbot.txt not found. Chatbot will not work.")

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

if len(training) > 0:
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
    print("✓ Chatbot model trained successfully!")
else:
    model = None
    print("Warning: No training data found for chatbot.")

def get_response(query):
    """Process input and predict category"""
    if model is None or len(stemmed_words) == 0:
        return "Sorry, the chatbot is not ready. Please try again later."
    
    row = [0] * len(stemmed_words)
    query = query.lower().translate(punct_dict)
    tokens = nltk.word_tokenize(query)
    tokens_stop = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens_stop]

    for stemmed_word in stemmed_tokens:
        for i, w in enumerate(stemmed_words):
            if w == stemmed_word:
                row[i] = 1

    response = model.predict(np.array([row]), verbose=0)
    results_index = np.argmax(response)
    tag = sorted_categories[results_index]
    
    # Get random response from category
    responses = [answers[i] for i, category in enumerate(categories) if category == tag]
    if responses:
        import random
        return random.choice(responses)
    else:
        return "I'm not sure how to respond to that. Can you ask something else?"

# ============ API ROUTES ============

# Mood Tracker Routes
@app.route('/api/mood', methods=['POST'])
def add_mood():
    """Add a new mood entry"""
    data = request.json
    mood_entry = MoodEntry(
        emoji=data.get('emoji'),
        label=data.get('label'),
        note=data.get('note', '')
    )
    db.session.add(mood_entry)
    db.session.commit()
    return jsonify({'success': True, 'id': mood_entry.id}), 201

@app.route('/api/mood', methods=['GET'])
def get_moods():
    """Get all mood entries"""
    moods = MoodEntry.query.order_by(MoodEntry.timestamp.desc()).all()
    return jsonify([mood.to_dict() for mood in moods]), 200

@app.route('/api/mood/<int:mood_id>', methods=['DELETE'])
def delete_mood(mood_id):
    """Delete a mood entry"""
    mood = MoodEntry.query.get(mood_id)
    if mood:
        db.session.delete(mood)
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found'}), 404

# Journal Routes
@app.route('/api/journal', methods=['POST'])
def add_journal():
    """Add a new journal entry"""
    data = request.json
    journal_entry = JournalEntry(
        title=data.get('title'),
        content=data.get('content')
    )
    db.session.add(journal_entry)
    db.session.commit()
    return jsonify({'success': True, 'id': journal_entry.id}), 201

@app.route('/api/journal', methods=['GET'])
def get_journal():
    """Get all journal entries"""
    entries = JournalEntry.query.order_by(JournalEntry.timestamp.desc()).all()
    return jsonify([entry.to_dict() for entry in entries]), 200

@app.route('/api/journal/<int:journal_id>', methods=['DELETE'])
def delete_journal(journal_id):
    """Delete a journal entry"""
    entry = JournalEntry.query.get(journal_id)
    if entry:
        db.session.delete(entry)
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found'}), 404

# Chatbot Routes
@app.route('/api/chat', methods=['POST'])
def chat():
    """Send a message to the chatbot"""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    # Get bot response
    bot_response = get_response(user_message)
    
    # Save to database
    chat_entry = ChatbotResponse(
        user_message=user_message,
        bot_response=bot_response
    )
    db.session.add(chat_entry)
    db.session.commit()
    
    return jsonify({
        'user_message': user_message,
        'bot_response': bot_response,
        'id': chat_entry.id
    }), 200

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history"""
    chats = ChatbotResponse.query.order_by(ChatbotResponse.timestamp.asc()).all()
    return jsonify([chat.to_dict() for chat in chats]), 200

@app.route('/api/chat/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat entry"""
    chat = ChatbotResponse.query.get(chat_id)
    if chat:
        db.session.delete(chat)
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found'}), 404

# Feedback Routes
@app.route('/api/feedback', methods=['POST'])
def add_feedback():
    """Add feedback form"""
    data = request.json
    feedback = FeedbackForm(
        name=data.get('name'),
        email=data.get('email'),
        feedback=data.get('feedback'),
        rating=data.get('rating', 0)
    )
    db.session.add(feedback)
    db.session.commit()
    return jsonify({'success': True, 'id': feedback.id}), 201

@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    """Get all feedback"""
    feedback_entries = FeedbackForm.query.order_by(FeedbackForm.timestamp.desc()).all()
    return jsonify([f.to_dict() for f in feedback_entries]), 200

@app.route('/api/feedback/<int:feedback_id>', methods=['DELETE'])
def delete_feedback(feedback_id):
    """Delete feedback"""
    feedback = FeedbackForm.query.get(feedback_id)
    if feedback:
        db.session.delete(feedback)
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found'}), 404

# App Feelings Routes
@app.route('/api/app-feelings', methods=['POST'])
def add_app_feelings():
    """Add app feelings form"""
    data = request.json
    app_feelings = AppFeelingsForm(
        app_name=data.get('app_name'),
        features_used=data.get('features_used'),
        feeling_about_app=data.get('feeling_about_app'),
        improvement_suggestions=data.get('improvement_suggestions'),
        would_recommend=data.get('would_recommend', False)
    )
    db.session.add(app_feelings)
    db.session.commit()
    return jsonify({'success': True, 'id': app_feelings.id}), 201

@app.route('/api/app-feelings', methods=['GET'])
def get_app_feelings():
    """Get all app feelings"""
    feelings = AppFeelingsForm.query.order_by(AppFeelingsForm.timestamp.desc()).all()
    return jsonify([f.to_dict() for f in feelings]), 200

@app.route('/api/app-feelings/<int:feelings_id>', methods=['DELETE'])
def delete_app_feelings(feelings_id):
    """Delete app feelings"""
    feelings = AppFeelingsForm.query.get(feelings_id)
    if feelings:
        db.session.delete(feelings)
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found'}), 404

# Statistics Routes
@app.route('/api/stats/dashboard', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    mood_count = MoodEntry.query.count()
    journal_count = JournalEntry.query.count()
    chat_count = ChatbotResponse.query.count()
    feedback_count = FeedbackForm.query.count()
    
    # Most common mood
    common_mood = None
    if mood_count > 0:
        moods = db.session.query(MoodEntry.label).all()
        mood_labels = [m[0] for m in moods]
        from collections import Counter
        mood_counts = Counter(mood_labels)
        common_mood = mood_counts.most_common(1)[0][0]
    
    return jsonify({
        'total_moods': mood_count,
        'total_journals': journal_count,
        'total_chats': chat_count,
        'total_feedback': feedback_count,
        'most_common_mood': common_mood
    }), 200

# Health check
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

# ============ CREATE DATABASE AND RUN ============

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✓ Database initialized!")
    
    app.run(debug=True, port=5000)
