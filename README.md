﻿# Chat with Multiple PDFs using Gemini
  This project is a Streamlit-based application designed to facilitate conversation with multiple PDF documents using Gemini, an AI model provided by Google GenerativeAI. It utilizes langchain for 
  text  processing, FAISS for vector storage, and PyPDF2 for PDF parsing.

### Introduction
  The Chat with Multiple PDFs project aims to provide a user-friendly interface for querying information across multiple PDF documents. It leverages Gemini, a conversational AI model, to answer user     questions based on the content extracted from uploaded PDF files.

### Installation
  To run the project locally, follow these steps:
  1. Clone the repository: `git clone https://github.com/your_username/your_repository.git`
  2. Navigate to the project directory: `cd your_repository`
  3. Install the required dependencies: `pip install -r requirements.txt`
  4. Set up the required environment variables. Obtain a Google API key and store it in a `.env` file: `GOOGLE_API_KEY=your_google_api_key`

### Usage
  Once the installation is complete, you can run the application using the following command: `streamlit run app.py`
  This command will start the Streamlit server, and you can access the application in your web browser at `http://localhost:8501`.

### How to Use
  1. **Upload PDF Files:** Use the provided interface to upload one or more PDF files containing the information you want to query.
  2. **Ask a Question:** Enter your question related to the content of the uploaded PDF files in the text input field.
  3. **Submit:** Click the "Submit" button to initiate the query process.
  4. **View Response:** The application will display the AI-generated response to your question.

### Workflow
  * The uploaded PDF files are processed to extract text content using PyPDF2.
  * The text content is then split into smaller chunks for efficient processing.
  * Text embeddings are generated using Google GenerativeAI.
  * The embeddings are stored in a FAISS index for similarity search.
  * User questions are matched against the indexed documents to find relevant information.
  * Gemini AI model is used to generate responses based on the matched documents and user queries.

## Dependencies
  * Streamlit
  * PyPDF2
  * Langchain
  * FAISS
  * Google GenerativeAI
  * dotenv


