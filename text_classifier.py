from transformers import pipeline
from transformers.tokenization_utils_base import TruncationStrategy

def classify_and_store_from_file(file_path):
    # Initialize the sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    
    # Open the input text file
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Clean up the line by removing leading/trailing whitespaces and newlines
            cleaned_line = line.strip()
            # Classify the cleaned line
            if '"' in cleaned_line:
                result = classifier(cleaned_line, truncation=TruncationStrategy.LONGEST_FIRST, max_length=512)
                # Extract label and score
                top_three_labels = []
                top_three_scores = []
                for entry in result[0]:
                    label = entry['label']
                    score = entry['score']
                    if len(top_three_labels) < 3:
                        top_three_labels.append(label)
                        top_three_scores.append(score)
                    else:
                        min_score_index = top_three_scores.index(min(top_three_scores))
                        if score > top_three_scores[min_score_index]:
                            top_three_labels[min_score_index] = label
                            top_three_scores[min_score_index] = score
                # Get the top-scoring label
                max_score_index = top_three_scores.index(max(top_three_scores))
                max_label = top_three_labels[max_score_index]
            else:
                # If no dialogue, set the label to "neutral"
                max_label = "neutral"
                all_label = 1.0
            
            # Create or open a text file based on the label
            output_file_path = f"new_{max_label}_text_2.txt"
            with open(output_file_path, 'a') as output_file:
                # Write the cleaned line and its sentiment score to the file
                if max_label != "neutral":
                    output_file.write(f"Text: {cleaned_line}\n")
                    output_file.write(f"Top Three Labels: {top_three_labels}\n")
                    output_file.write(f"Top Three Scores: {top_three_scores}\n\n")
                else:
                    output_file.write(f"Text: {cleaned_line}\n")
                    output_file.write(f"Label: {max_label}\n")
                    output_file.write(f"Score: {all_label}\n\n")

# Execute function
input_file_path = "Input file path here"
classify_and_store_from_file(input_file_path)
