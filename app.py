"""
Document Q&A Application with PDF Citation Highlighting

This Streamlit application allows users to upload PDF documents and ask questions
about their content. The AI assistant (Claude) provides conversational answers with
numbered citations that can be traced back to exact quotes in the source document.

Key Features:
- PDF document upload and viewing
- Conversational Q&A with citation support
- Visual highlighting of source quotes on PDF pages
- Citation numbers linking answers to verifiable sources

Author: Paul Morat
Date: 05.02.2026
Assignment for kaiko.ai
"""

import base64
import os
import re
from io import BytesIO

import fitz  # PyMuPDF - for PDF text search and coordinate extraction
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from PIL import ImageDraw

# -------------------------
# Constants & Configuration
# -------------------------

# PDF rendering settings
PDF_DPI = 150  # Resolution for PDF to image conversion

# Highlight styling
HIGHLIGHT_COLOR = (255, 193, 7, 255)  # Amber/Yellow color (RGBA)
HIGHLIGHT_WIDTH = 5  # Thickness of underline in pixels
HIGHLIGHT_OFFSET = 2  # Pixels below text baseline

# Text search settings
SEARCH_CHAR_LIMIT = 50  # Characters to try if full quote not found
SEARCH_WORD_LIMIT = 6  # Words to try if character search fails

# API settings
MAX_RESPONSE_TOKENS = 2048  # Maximum tokens in Claude's response
MODEL_NAME = "claude-sonnet-4-5"  # Claude model to use

# -------------------------
# Environment Setup
# -------------------------

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Please set ANTHROPIC_API_KEY in your .env file")
    st.stop()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for source citation styling
st.markdown("""
<style>
.source-highlight {
    background-color: #FFF3CD;
    border-left: 4px solid #FFC107;
    padding: 10px;
    margin: 8px 0;
    font-size: 0.9em;
    border-radius: 4px;
}
.source-title {
    font-weight: bold;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State Initialization
# -------------------------

# Define default values for all session state variables
SESSION_DEFAULTS = {
    "messages": [],           # Chat history (user and assistant messages)
    "pdf_bytes": None,        # Raw PDF file content
    "pdf_images": [],         # PDF pages converted to PIL images
    "highlight_coords": [],   # Bounding boxes for current highlight
    "selected_page": 1,       # Currently displayed page (1-indexed)
    "current_quote": None,    # Quote text currently being highlighted
}

# Initialize any missing session state variables
for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# -------------------------
# System Prompt
# -------------------------

SYSTEM_PROMPT = """You are a knowledgeable research assistant having a conversation about a document. 

YOUR TONE:
- Be conversational and natural, like explaining findings to a colleague
- Use clear, accessible language - avoid overly formal or robotic phrasing
- You can use phrases like "Interestingly...", "The key finding here is...", "What stands out is..."
- Structure your response with natural flow, not rigid numbered lists
- Feel free to highlight what's most important or surprising

CITATION FORMAT:
- After each factual claim in your answer, add a citation number in brackets like [01], [02], [03]
- Number citations in order starting from [01]
- The citation number must match the source number in the SOURCES section

RESPONSE FORMAT:
[Your natural, conversational answer with citation numbers [01], [02], etc. after each factual claim]

SOURCES:
[01] Short label: "Exact quote from document" — Page X
[02] Short label: "Exact quote from document" — Page X
[03] Short label: "Exact quote from document" — Page X

EXAMPLE:
The study reveals some fascinating findings about AI in clinical settings. The most striking result is that more detailed explanations didn't actually improve trust or performance [01]. In fact, simpler explanations worked better overall [02]. 

What's particularly noteworthy is how participants responded differently based on their background. While the AI generally helped with diagnostic accuracy, demographic factors like age and experience influenced self-reported measures but not actual behavioral outcomes [03].

SOURCES:
[01] Explanation paradox: "increasing levels of explanations do not always improve trust or diagnosis performance" — Page 1
[02] Simpler is better: "no explanation intervention with the simplest design would be the ideal choice" — Page 9
[03] Demographics effect: "demographic factors influenced self-reported measures but not behavioral assessments" — Page 8

STRICT RULES:
1. Every factual claim MUST have a citation number [01], [02], etc.
2. Citation numbers in your answer MUST match the source numbers
3. Quotes must be EXACT text from the document
4. Always include the page number
5. If something spans pages, write "Page X-Y"
6. Use two-digit format: [01], [02], ... [09], [10], etc."""

# -------------------------
# Core Functions
# -------------------------

@st.cache_data
def load_pdf_images(file_bytes: bytes) -> list:
    """
    Convert PDF file bytes to a list of PIL images for display.
    
    Uses caching to avoid re-processing the same PDF multiple times.
    
    Args:
        file_bytes: Raw bytes of the PDF file
        
    Returns:
        List of PIL Image objects, one per page
    """
    return convert_from_bytes(file_bytes, dpi=PDF_DPI)


def extract_pdf_coordinates(pdf_bytes: bytes, page_num: int, text_snippet: str) -> list:
    """
    Find the bounding box coordinates of a text snippet on a specific PDF page.
    
    Uses PyMuPDF to search for text and returns coordinates that can be used
    to draw highlights on the rendered page image.
    
    The function tries multiple search strategies:
    1. Full text search
    2. First N characters (if full text not found)
    3. First N words (if character search fails)
    
    Args:
        pdf_bytes: Raw bytes of the PDF file
        page_num: Page number to search (1-indexed)
        text_snippet: Text to find on the page
        
    Returns:
        List of bounding boxes as [x0, y0, x1, y1] coordinates
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Validate page number
    if page_num < 1 or page_num > len(doc):
        return []

    page = doc[page_num - 1]  # Convert to 0-indexed
    
    # Clean the search text
    clean_text = text_snippet.replace("\u0002", "").strip()
    
    # Strategy 1: Try to find the full text
    quads = page.search_for(clean_text)
    
    # Strategy 2: Try first N characters if full text not found
    if not quads and len(clean_text) > SEARCH_CHAR_LIMIT:
        quads = page.search_for(clean_text[:SEARCH_CHAR_LIMIT])
    
    # Strategy 3: Try first N words if character search fails
    if not quads:
        words = clean_text.split()[:SEARCH_WORD_LIMIT]
        if words:
            quads = page.search_for(" ".join(words))
    
    return [list(q) for q in quads]


def draw_highlights_on_image(pil_image, coords: list, pdf_w: float, pdf_h: float):
    """
    Draw yellow underline highlights on a PIL image based on PDF coordinates.
    
    Converts PDF coordinate space to image pixel space and draws underlines
    beneath the specified text regions.
    
    Args:
        pil_image: PIL Image object to draw on
        coords: List of bounding boxes [x0, y0, x1, y1] in PDF coordinates
        pdf_w: Width of the PDF page in points
        pdf_h: Height of the PDF page in points
        
    Returns:
        PIL Image with highlights drawn
    """
    if not coords:
        return pil_image

    # Create RGBA overlay for smooth drawing
    overlay = pil_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    
    img_w, img_h = pil_image.size
    
    # Calculate scale factors from PDF points to image pixels
    scale_x = img_w / pdf_w
    scale_y = img_h / pdf_h

    for x0, y0, x1, y1 in coords:
        # Scale coordinates from PDF space to image space
        sx0 = x0 * scale_x
        sx1 = x1 * scale_x
        sy1 = y1 * scale_y
        
        # Draw underline slightly below the text baseline
        line_y = sy1 + HIGHLIGHT_OFFSET
        
        draw.line(
            [(sx0, line_y), (sx1, line_y)],
            fill=HIGHLIGHT_COLOR,
            width=HIGHLIGHT_WIDTH
        )
    
    return overlay.convert("RGB")


def parse_response(response_text: str) -> tuple:
    """
    Parse Claude's response to separate the answer from sources.
    
    Expects response format:
        [Answer text with citations]
        
        SOURCES:
        [01] Label: "Quote" — Page X
        [02] Label: "Quote" — Page Y
    
    Args:
        response_text: Full response text from Claude
        
    Returns:
        Tuple of (answer_text, list_of_sources)
    """
    if "SOURCES:" in response_text:
        parts = response_text.split("SOURCES:")
        answer = parts[0].replace("ANSWER:", "").strip()
        sources_text = parts[1].strip()
        
        # Parse individual source lines
        sources = []
        for line in sources_text.split("\n"):
            line = line.strip()
            # Accept lines starting with citation number or dash
            if line.startswith("-") or line.startswith("["):
                if line.startswith("-"):
                    line = line[1:].strip()
                sources.append(line)
        
        return answer, sources
    
    # No SOURCES section found - return full text as answer
    return response_text, []


def extract_quote_and_page(source: str) -> tuple:
    """
    Extract the quote text and page number from a source citation string.
    
    Handles both single pages ("Page 5") and page ranges ("Page 3-4").
    
    Args:
        source: Source string like '[01] Label: "quote text" — Page 5'
        
    Returns:
        Tuple of (quote_text, page_number_or_tuple)
        Page can be int (single page) or tuple (start, end) for ranges
    """
    # Extract quoted text
    quote_match = re.search(r'"([^"]+)"', source)
    quote = quote_match.group(1) if quote_match else None
    
    # Check for page range (e.g., "Page 3-4")
    range_match = re.search(r'Page\s*(\d+)\s*-\s*(\d+)', source)
    if range_match:
        return quote, (int(range_match.group(1)), int(range_match.group(2)))
    
    # Check for single page (e.g., "Page 5")
    single_match = re.search(r'Page\s*(\d+)', source)
    if single_match:
        return quote, int(single_match.group(1))
    
    return quote, None


def extract_citation_number(source: str) -> str:
    """
    Extract the citation number from a source string.
    
    Args:
        source: Source string like '[01] Label: "quote" — Page 5'
        
    Returns:
        Citation number as string (e.g., "01") or None if not found
    """
    match = re.match(r'\[(\d+)\]', source)
    return match.group(1) if match else None


# -------------------------
# Sidebar: Document Upload
# -------------------------

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        
        # Check if this is a new file (different from currently loaded)
        if st.session_state.pdf_bytes != file_bytes:
            # Load new document and reset state
            st.session_state.pdf_bytes = file_bytes
            st.session_state.pdf_images = load_pdf_images(file_bytes)
            st.session_state.messages = []
            st.session_state.highlight_coords = []
            st.session_state.selected_page = 1
            st.session_state.current_quote = None
            st.success(f"✅ Loaded {len(st.session_state.pdf_images)} pages")
    
    # Chat management buttons
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.highlight_coords = []
        st.session_state.current_quote = None
        st.rerun()
    
    if st.button("🔄 Clear Highlights"):
        st.session_state.highlight_coords = []
        st.session_state.current_quote = None
        st.rerun()


# -------------------------
# Main Content Area
# -------------------------

# Show welcome message if no document loaded
if not st.session_state.pdf_bytes:
    st.title("📄 Document Q&A")
    st.info("👈 Please upload a PDF document to get started")
    st.stop()

st.title("📄 Document Q&A")

# Create two-column layout: Chat on left, Document viewer on right
col_chat, col_doc = st.columns([1, 1])

# -------------------------
# Left Column: Chat Interface
# -------------------------

with col_chat:
    st.subheader("💬 Chat")
    
    # Dynamic chat container height based on message count
    chat_height = 200 if len(st.session_state.messages) == 0 else 500
    chat_container = st.container(height=chat_height)
    
    with chat_container:
        # Display all messages in chat history
        for msg_idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources expander for assistant messages with citations
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources"):
                        for src_idx, source in enumerate(message["sources"]):
                            quote, page = extract_quote_and_page(source)
                            citation_num = extract_citation_number(source)
                            
                            # Display formatted source citation
                            st.markdown(
                                f'<div class="source-highlight">{source}</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Add button to highlight quote in document viewer
                            if page and quote:
                                if isinstance(page, tuple):
                                    start_page, end_page = page
                                    button_label = f"🔍 Show on Page {start_page}-{end_page}"
                                    go_to_page = start_page
                                else:
                                    button_label = f"🔍 Show on Page {page}"
                                    go_to_page = page
                                
                                if st.button(button_label, key=f"goto_{msg_idx}_{src_idx}"):
                                    # Update state to show highlight on selected page
                                    st.session_state.selected_page = go_to_page
                                    st.session_state.current_quote = quote
                                    
                                    # Find coordinates for highlighting
                                    coords = extract_pdf_coordinates(
                                        st.session_state.pdf_bytes,
                                        go_to_page,
                                        quote
                                    )
                                    st.session_state.highlight_coords = coords
                                    st.rerun()
                            
                            st.write("")  # Spacing between sources

    # Chat input field
    user_input = st.chat_input("Ask a question about your document...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Prepare PDF for Claude API (base64 encoded)
        pdf_b64 = base64.b64encode(st.session_state.pdf_bytes).decode()
        
        # Construct message payload with document and question
        message_payload = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64
                },
                "title": "Uploaded Document"
            },
            {
                "type": "text",
                "text": user_input
            }
        ]

        # Call Claude API
        with st.spinner("Analyzing document..."):
            try:
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": message_payload}]
                )
                
                # Parse response into answer and sources
                full_response = response.content[0].text
                answer, sources = parse_response(full_response)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}. Please try again.")
                # Remove the user message that caused the error
                st.session_state.messages.pop()


# -------------------------
# Right Column: Document Viewer
# -------------------------

with col_doc:
    st.subheader("📄 Document Viewer")
    
    total_pages = len(st.session_state.pdf_images)
    
    # Page navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Prev"):
            if st.session_state.selected_page > 1:
                st.session_state.selected_page -= 1
                st.session_state.highlight_coords = []
                st.session_state.current_quote = None
                st.rerun()
    
    with col2:
        new_page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=st.session_state.selected_page - 1,
            format_func=lambda x: f"Page {x} of {total_pages}",
            label_visibility="collapsed"
        )
        if new_page != st.session_state.selected_page:
            st.session_state.selected_page = new_page
            st.session_state.highlight_coords = []
            st.session_state.current_quote = None
            st.rerun()
    
    with col3:
        if st.button("Next →"):
            if st.session_state.selected_page < total_pages:
                st.session_state.selected_page += 1
                st.session_state.highlight_coords = []
                st.session_state.current_quote = None
                st.rerun()
    
    # Display highlight status messages
    if st.session_state.current_quote and st.session_state.highlight_coords:
        st.success("📍 Quote highlighted on this page")
        with st.expander("🔍 Looking for this quote:"):
            st.write(f'"{st.session_state.current_quote}"')
    elif st.session_state.current_quote and not st.session_state.highlight_coords:
        st.warning("⚠️ Could not locate exact quote on this page")
        with st.expander("🔍 Looking for this quote:"):
            st.write(f'"{st.session_state.current_quote}"')
    
    # Get current page image
    current_img = st.session_state.pdf_images[st.session_state.selected_page - 1]
    
    # Apply highlights if coordinates exist
    if st.session_state.highlight_coords:
        doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
        page_obj = doc[st.session_state.selected_page - 1]
        
        current_img = draw_highlights_on_image(
            current_img,
            st.session_state.highlight_coords,
            page_obj.rect.width,
            page_obj.rect.height
        )
    
    # Display the PDF page
    st.image(current_img, use_container_width=True)

