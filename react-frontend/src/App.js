import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("");
  const [pptUrl, setPptUrl] = useState("");
  const [message, setMessage] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !query) {
      setMessage("Please select a file and enter a query.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("query", query);

    try {
      setMessage("Processing...");
      const response = await axios.post("http://127.0.0.1:5000/", formData);
      if (response.data.downloadUrl) {
        setPptUrl(response.data.downloadUrl);
        setMessage("Processing complete! Click below to download.");
      } else {
        setMessage("Processing complete! But no file found.");
      }
    } catch (error) {
      console.error(error);
      setMessage("An error occurred while processing the file.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Project Demo</h1>
        <form onSubmit={handleSubmit}>
          <input type="file" accept=".xlsx, .xls" onChange={handleFileChange} />
          <input type="text" placeholder="Enter your query" value={query} onChange={handleQueryChange} />
          <button type="submit">Generate PPT</button>
        </form>
        {message && <p>{message}</p>}
        {pptUrl && <a href={pptUrl} download>Download PPT</a>}
      </header>
    </div>
  );
}

export default App;
