# Smart PDF Chatbot

A RAG-based chatbot for chatting with multiple PDFs using Streamlit and LangChain.

## Features

- Upload multiple PDF documents
- Ask questions and get AI-powered answers based on the PDF content
- Conversation history with CSV download
- Clean, modular architecture

## Project Structure

```
Smart_PDF_Chatbot/
├── app/
│   └── main.py              # Main Streamlit application
├── services/
│   ├── pdf_processor.py     # PDF text extraction
│   └── rag_service.py       # RAG pipeline components
├── ui/
│   └── chat_ui.py           # UI components for chat display
├── utils/
│   └── helpers.py           # Helper functions and validations
├── config/
│   └── settings.py          # Configuration and environment variables
├── data/                    # Directory for cached data (e.g., vectorstores)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Setup

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Google API key: `GOOGLE_API_KEY=your_key_here`

6. Run the application:
   ```bash
   streamlit run app/main.py
   ```

## Usage

- Upload PDF files using the sidebar uploader.
- Enter your Google API key.
- Ask questions about the uploaded PDFs.
- View conversation history and download as CSV.

## Improvements Made

- Modular architecture for better maintainability
- Caching for performance optimization
- Error handling and input validation
- Environment variable management for security
- Clean separation of concerns (UI, services, config)