from flask import Flask, request, jsonify
from transformers import pipeline
import builtins
from waitress import serve
import os
import logging 
from werkzeug.exceptions import InternalServerError
import config

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

    sentiment_summary = ', '.join([f"{result['label']}: {result['score']:.2f}%" for result in normalized_results])

    return sentiment_summary

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Emotion Sentiment Analysis API!"

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    try:
        content = request.json
        if not content or 'text' not in content:
            return jsonify({"error": "No text provided"}), 400

        model_id = "j-hartmann/emotion-english-distilroberta-base"
        labels_to_keep = {"joy", "anger", "neutral", "sadness"}

        results = emotionSentiment(model_id, labels_to_keep, text=content['text'])
        return jsonify({'sentiment': results})
    except Exception as e:
        logging.exception("An error occurred in analyze_emotion: %s", str(e))
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/api/lyrics', methods=['POST'])
def submit_lyrics():
    try:
        content = request.json
        if not content or 'user_id' not in content or 'lyrics' not in content:
            return jsonify({"error": "Missing user_id or lyrics"}), 400

        user_id = content['user_id']
        lyrics = content['lyrics']

        # Additional input validation
        if not isinstance(lyrics, str) or len(lyrics.strip()) == 0:
            return jsonify({"error": "Invalid lyrics format"}), 400

        logging.info(f"Processing lyrics for user_id: {user_id}")

        results = emotionSentiment(config.MODEL_ID, config.LABELS_TO_KEEP, text=lyrics)

        logging.info(f"Analysis completed for user_id: {user_id}")

        return jsonify({
            "message": "Lyrics received and analyzed successfully",
            "user_id": user_id,
            "lyrics": lyrics,
            "analysis_results": results
        })
    except Exception as e:
        logging.exception(f"An error occurred in submit_lyrics for user_id {content.get('user_id', 'unknown')}: {str(e)}")
        return jsonify({"error": "An internal error occurred during lyrics analysis"}), 500


def test_emotionSentiment():
    model_id = "j-hartmann/emotion-english-distilroberta-base"
    labels_to_keep = {"joy", "anger", "neutral", "sadness"}
    test_text = "I am feeling very happy today!"
    
    try:
        result = emotionSentiment(model_id, labels_to_keep, test_text)
        print(f"Test result: {result}")
        return True
    except Exception as e:
        print(f"Error in emotionSentiment: {str(e)}")
        return False

if __name__ == '__main__':
    if test_emotionSentiment():
        print("emotionSentiment function test passed")
    else:
        print("emotionSentiment function test failed")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Starting server on port {config.PORT}")  # Add this line
    serve(app, host='0.0.0.0', port=config.PORT, threads=config.THREADS)

