from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
from components.file_upload import upload_file
from components.preprocess import preprocess_data
from components.embeddings import generate_tags_with_context, get_bert_embeddings
from components.query_analysis import (
    apply_filters,
    match_query_to_tags,
    extract_conditions_from_query,
    extract_entities,
    identify_intent,
    filter_rows_based_on_common_keywords
)
from components.ppt_generation import perform_descriptive_statistics, sanitize_filename

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(os.getcwd(), "outputs")

# Create folders if they do not exist
for folder in [app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"]]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route("/", methods=["POST"])
def process_file():
    try:
        if "file" not in request.files or request.form.get("query", "") == "":
            return jsonify({"error": "Please upload an Excel file and enter a query."}), 400

        # Step 1: Upload file
        file, query = upload_file(request, app)

        # Step 2: Preprocess file
        data = preprocess_data(file)

        # Step 3: Generate and store embeddings
        tag_embeddings = generate_tags_with_context(data)

        # Step 4: Query analysis
        conditions = extract_conditions_from_query(query)
        entities = extract_entities(query)
        user_intent = identify_intent(query)
        matched_columns, matched_values = match_query_to_tags(query, tag_embeddings, threshold=0.7)

        # Apply filters
        filtered_data = apply_filters(data, matched_columns, conditions, matched_values, query)

        # Step 5: PowerPoint generation
        sanitized_query = sanitize_filename(query)
        ppt_filename = f"{sanitized_query}.pptx"
        ppt_path = os.path.join(app.config["OUTPUT_FOLDER"], ppt_filename)
        perform_descriptive_statistics(filtered_data, query, ppt_path, sanitized_query)

        download_url = f"http://127.0.0.1:5000/download/{ppt_filename}"
        return jsonify({"downloadUrl": download_url})

    except Exception as e:
        error_message = str(e)
        traceback_details = traceback.format_exc()
        print(f"Error: {error_message}\n{traceback_details}")  # error logging
        return jsonify({"error": f"Error: {error_message}"}), 500

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
