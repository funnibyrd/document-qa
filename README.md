# Document Q&A with Citation Highlighting
git remote add origin https://github.com/YOUR_USERNAME/document-qa.git
git branch -M main
git push -u origin main

## Features

- **PDF Upload**: Upload any PDF document for analysis
- **Conversational Q&A**: Ask natural language questions about the document
- **Numbered Citations**: Every claim is linked to a source [01], [02], etc.
- **Source Verification**: Click any citation to see the exact quote
- **PDF Highlighting**: Visual highlighting of source quotes on the actual PDF page

## Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Claude Sonnet 4.5 (Anthropic API)
- **PDF Processing**: PyMuPDF (fitz), pdf2image
- **Image Processing**: Pillow

## Setup

1. Clone the repository
2. Install dependencies:
```bash
   pip install streamlit anthropic python-dotenv pdf2image PyMuPDF pillow
   brew install poppler  # macOS only
```
3. Create a `.env` file with your API key:
```
   ANTHROPIC_API_KEY=your_key_here
```
4. Run the app:
```bash
   streamlit run app.py
```

## Usage

1. Upload a PDF document using the sidebar
2. Ask questions in the chat (e.g., "What were the primary outcomes?")
3. View numbered citations in the answer
4. Click "📚 View Sources" to see exact quotes
5. Click "🔍 Show on Page X" to highlight the quote in the PDF viewer

