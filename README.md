# Emotion_text_classifier_hartmann

This program parses through any text files given and uses J. Hartmann's Emotion English DistilroBERTa-base classifier to categorize each line into one of 6 emotions (including a neutral state). Returns ~76% overall accuracy in correct classification.

Requires to pip install transformers to use

Only takes text files and returns a 7 text files that include the line, top three emotion labels, and the scores for those emotion labels.



References:
Hartmann, J. (2022). Emotion english distilroberta-base. Retrieved 2024-05, from https://
huggingface.co/j-hartmann/emotion-english-distilroberta-base/
