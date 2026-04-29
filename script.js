// API Base URL
const API_BASE = 'http://localhost:5000/api';

// ============ SECTION NAVIGATION ============

function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
    event.target.classList.add('active');
    window.scrollTo(0, 0);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadMoodData();
    loadJournalData();
    loadChatHistory();
    updateMoodStats();
});

// ============ MOOD TRACKER ============

let currentMood = null;

function addMood(emoji, label) {
    currentMood = { emoji, label };
}

async function saveMoodEntry() {
    if (!currentMood) {
        alert('Please select a mood first!');
        return;
    }

    const note = document.getElementById('mood-note').value;
    const date = new Date().toLocaleDateString();

    try {
        const response = await fetch(`${API_BASE}/mood`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                emoji: currentMood.emoji,
                label: currentMood.label,
                note: note,
                date: date
            })
        });

        if (response.ok) {
            document.getElementById('mood-note').value = '';
            currentMood = null;
            loadMoodData();
            updateMoodStats();
            alert('✓ Mood entry saved!');
        }
    } catch (error) {
        console.error('Error saving mood:', error);
    }
}

async function loadMoodData() {
    try {
        const response = await fetch(`${API_BASE}/mood`);
        const moods = await response.json();
        const moodLog = document.getElementById('mood-log');

        if (moods.length === 0) {
            moodLog.innerHTML = '<p class="empty-state">No mood entries yet. Start tracking!</p>';
            return;
        }

        moodLog.innerHTML = moods.reverse().map(entry => `
            <div class="mood-entry">
                <div class="mood-entry-header">
                    <span class="mood-entry-emoji">${entry.emoji}</span>
                    <span class="mood-entry-label">${entry.label}</span>
                    <span class="mood-entry-date">${entry.timestamp}</span>
                </div>
                ${entry.note ? `<div class="mood-entry-note"><strong>Note:</strong> ${entry.note}</div>` : ''}
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading moods:', error);
    }
}

async function updateMoodStats() {
    try {
        const response = await fetch(`${API_BASE}/mood`);
        const moods = await response.json();

        document.getElementById('total-entries').textContent = moods.length;

        if (moods.length > 0) {
            const moodCounts = {};
            moods.forEach(entry => {
                moodCounts[entry.label] = (moodCounts[entry.label] || 0) + 1;
            });

            const mostCommon = Object.keys(moodCounts).reduce((a, b) =>
                moodCounts[a] > moodCounts[b] ? a : b
            );

            const commonEmoji = moods.find(e => e.label === mostCommon)?.emoji || '';
            document.getElementById('common-mood').textContent = `${commonEmoji} ${mostCommon}`;
        }
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// ============ JOURNAL ============

async function saveJournalEntry() {
    const title = document.getElementById('journal-title').value.trim();
    const content = document.getElementById('journal-content').value.trim();

    if (!title || !content) {
        alert('Please add both title and content!');
        return;
    }

    const date = new Date().toLocaleDateString();

    try {
        const response = await fetch(`${API_BASE}/journal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: title,
                content: content,
                date: date
            })
        });

        if (response.ok) {
            document.getElementById('journal-title').value = '';
            document.getElementById('journal-content').value = '';
            loadJournalData();
            alert('✓ Journal entry saved!');
        }
    } catch (error) {
        console.error('Error saving journal:', error);
    }
}

async function loadJournalData() {
    try {
        const response = await fetch(`${API_BASE}/journal`);
        const entries = await response.json();
        const journalLog = document.getElementById('journal-log');

        if (entries.length === 0) {
            journalLog.innerHTML = '<p class="empty-state">No journal entries yet. Start writing!</p>';
            return;
        }

        journalLog.innerHTML = entries.reverse().map(entry => `
            <div class="journal-entry">
                <div class="journal-entry-title">${entry.title}</div>
                <div class="journal-entry-date">${entry.timestamp}</div>
                <div class="journal-entry-content">${entry.content}</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading journal:', error);
    }
}

// ============ CHATBOT ============

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();

    if (!query) return;

    // Add user message to chat
    const chatMessages = document.getElementById('chat-messages');
    const userMessage = document.createElement('div');
    userMessage.className = 'chat-message user-message';
    userMessage.innerHTML = `<p>${query}</p>`;
    chatMessages.appendChild(userMessage);

    input.value = '';

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        // Add bot response
        const botMessage = document.createElement('div');
        botMessage.className = 'chat-message bot-message';
        botMessage.innerHTML = `<p>${data.response}</p>`;
        chatMessages.appendChild(botMessage);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Load chat history
        loadChatHistory();
    } catch (error) {
        console.error('Error sending message:', error);
    }
}

function handleChatKeypress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

async function loadChatHistory() {
    try {
        const response = await fetch(`${API_BASE}/chat/history`);
        const history = await response.json();
        const historyLog = document.getElementById('chat-history-log');

        if (history.length === 0) {
            historyLog.innerHTML = '<p class="empty-state">No conversations yet.</p>';
            return;
        }

        historyLog.innerHTML = history.slice(0, 10).map(entry => `
            <div class="chat-history-item">
                <strong>You:</strong> ${entry.user_query.substring(0, 50)}...
                <strong>Bot:</strong> ${entry.bot_response.substring(0, 50)}...
                <span class="history-time">${entry.timestamp}</span>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// ============ MEDITATION ============

let meditationTimer = null;
let meditationTime = 0;

function startMeditation(type, title, description) {
    document.getElementById('meditation-title').textContent = title;
    document.getElementById('meditation-instruction').textContent = description;
    document.getElementById('meditation-player').style.display = 'flex';

    const durations = {
        'breathing': 300,
        'body-scan': 600,
        'mindfulness': 480,
        'loving-kindness': 600,
        'sleep': 900,
        'stress-relief': 420
    };

    meditationTime = durations[type] || 300;
    updateMeditationTimer();
}

function playMeditation() {
    if (meditationTimer) return;

    meditationTimer = setInterval(() => {
        meditationTime--;
        updateMeditationTimer();

        if (meditationTime <= 0) {
            clearInterval(meditationTimer);
            alert('✓ Great job! Session complete!');
            closeMeditation();
        }
    }, 1000);
}

function pauseMeditation() {
    if (meditationTimer) {
        clearInterval(meditationTimer);
        meditationTimer = null;
    }
}

function updateMeditationTimer() {
    const minutes = Math.floor(meditationTime / 60);
    const seconds = meditationTime % 60;
    document.getElementById('meditation-timer').textContent =
        `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function closeMeditation() {
    pauseMeditation();
    document.getElementById('meditation-player').style.display = 'none';
    meditationTime = 0;
}

// ============ FEEDBACK FORMS ============

async function submitFeedback(event) {
    event.preventDefault();

    const name = document.getElementById('feedback-name').value;
    const email = document.getElementById('feedback-email').value;
    const message = document.getElementById('feedback-message').value;
    const rating = document.querySelector('input[name="rating"]:checked').value;

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                email: email,
                message: message,
                rating: parseInt(rating)
            })
        });

        if (response.ok) {
            alert('✓ Thank you for your feedback!');
            event.target.reset();
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
    }
}

async function submitAppFeelings(event) {
    event.preventDefault();

    const helpful = document.querySelector('input[name="helpful"]:checked').value;
    const favorite = document.getElementById('favorite-feature').value;
    const improve = document.getElementById('improve-text').value;
    const continueUsing = document.querySelector('input[name="continue"]:checked').value;

    try {
        const response = await fetch(`${API_BASE}/app-feelings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                helpful: helpful,
                favorite_feature: favorite,
                improve: improve,
                continue_using: continueUsing
            })
        });

        if (response.ok) {
            alert('✓ Thanks for the feedback!');
            event.target.reset();
        }
    } catch (error) {
        console.error('Error submitting app feelings:', error);
    }
}

// ============ UTILITIES ============

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('✓ Copied to clipboard!');
    });
}
