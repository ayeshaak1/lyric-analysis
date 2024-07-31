from flask import Flask, request, jsonify
from transformers import pipeline
import builtins
from waitress import serve
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

app = Flask(__name__)


def emotionSentiment(model_id, labels_to_keep, text):
    classifier = pipeline("text-classification", model=model_id, return_all_scores=True)
    results = classifier(text)

    filtered_results = [result for result in results[0] if result['label'] in labels_to_keep]

    total_score = builtins.sum(result['score'] for result in filtered_results)
    normalized_results = [
        {'label': result['label'], 'score': (result['score'] / total_score) * 100}
        for result in filtered_results
    ]

    # Create a summary of the sentiment analysis
    sentiment_summary = ', '.join([f"{result['label']}: {result['score']:.2f}%" for result in normalized_results])

    return sentiment_summary


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Emotion Sentiment Analysis API!"


@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    content = request.json
    if not content or 'text' not in content:
        return jsonify({"error": "No text provided"}), 400

    model_id = "j-hartmann/emotion-english-distilroberta-base"
    labels_to_keep = {"joy", "anger", "neutral", "sadness"}

    results = emotionSentiment(model_id, labels_to_keep, text=content['text'])
    return jsonify({'sentiment': results})


@app.route('/api/lyrics', methods=['POST'])
def submit_lyrics():
    content = request.json
    if not content or 'user_id' not in content or 'lyrics' not in content:
        return jsonify({"error": "Missing user_id or lyrics"}), 400

    user_id = content['user_id']
    lyrics = content['lyrics']

    # Here you can save the lyrics to a database if needed

    # Perform emotion sentiment analysis on the lyrics
    model_id = "j-hartmann/emotion-english-distilroberta-base"
    labels_to_keep = {"joy", "anger", "neutral", "sadness"}
    results = emotionSentiment(model_id, labels_to_keep, text=lyrics)

    return jsonify({
        "message": "Lyrics received and analyzed successfully",
        "user_id": user_id,
        "analysis_results": results
    })


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=50100, threads=2)
