import os

def upload_file(request, app):
    if "file" not in request.files or request.form.get("query", "") == "":
        raise ValueError("Please upload an Excel file and enter a query.")
    
    file = request.files["file"]
    query = request.form.get("query")
    
    if file.filename == "":
        raise ValueError("No selected file")

    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    return file_path, query
