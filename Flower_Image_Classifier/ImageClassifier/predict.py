import data_processor
import checkpoint_processor
import argparse
import json

# Setup argparse to handle command line arguments
parser = argparse.ArgumentParser(description='Predict the type of flower from an image.')
parser.add_argument('image_path', help='Path to image file')
parser.add_argument('checkpoint', help='Path to a saved model checkpoint')
parser.add_argument('--top_k', type=int, default=3, help='Default is 3. Return top k most likely classes')
parser.add_argument('--category_names', default='cat_to_name.json',help='Default is cat_to_name.json. Use a mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available.')

# Parse the arguments
args = parser.parse_args()

# Load the model from checkpoint
model = checkpoint_processor.load_checkpoint(args.checkpoint)

# Process the input image
processed_image = data_processor.process_image(args.image_path)
probs, classes_indices = checkpoint_processor.predict_image(processed_image, model, args.top_k)

# Load the category names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Map the predicted classes indices to actual flower names
predicted_classes = [cat_to_name[str(index)] for index in classes_indices]

# Print the results
print("Probabilities: " + ", ".join(["{:.2f}%".format(prob * 100) for prob in probs]))
print("Classes: " + ", ".join(predicted_classes))