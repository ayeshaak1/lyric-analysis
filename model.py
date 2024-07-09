import transformers 
from transformers import pipeline 
import builtins

def emotionSentiment(model_id, labels_to_keep, text=None):
    if text is None:
        text = input("Enter the text to classify: ")
    
    classifier = pipeline("text-classification", model=model_id, return_all_scores=True)
    results = classifier(text)
    
    filtered_results = [result for result in results[0] if result['label'] in labels_to_keep]
    
    total_score = builtins.sum(result['score'] for result in filtered_results)  
    normalized_results = [
        {'label': result['label'], 'score': (result['score'] / total_score) * 100}
        for result in filtered_results
    ]
    
    return normalized_results


model_id = "j-hartmann/emotion-english-distilroberta-base"
labels_to_keep = {"joy", "anger", "neutral", "sadness"}

text = input("Enter lyrics that define your current mood: ")
filtered_results_with_text = emotionSentiment(model_id, labels_to_keep, text=text)
print(filtered_results_with_text)