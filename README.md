# SentimentSense: Neuro-Inclusive Communication Layer

SentimentSense is an assistive communication "prosthetic" designed specifically for individuals with **ADHD, Autism, and Neurodivergent (ND)** traits. It translates the unspoken rules of digital communication into clear, actionable, and low-anxiety insights.

## ✨ Core Features

-   **Social Forensics (Ambiguity Matrix)**: Decodes subtext and identifies the "Intent-Impact Trap" in vague messages.
-   **Executive Function Support (Step Breakdown)**: Deconstructs overwhelming walls-of-text into clear, prioritized checklists.
-   **Spoon Theory Integration**: Tailors response drafts based on your current energy level (Low Spoons = Minimal Masking).
-   **RSD Shield (Future Simulator)**: Predicts how the sender will react to your draft so you can "A/B test" social outcomes.
-   **Knowledge Base (RAG)**: Grounded in neurodivergent patterns like PDA, Time Blindness, and the Double Empathy Problem.

## 🛠️ Tech Stack

-   **Backend**: FastAPI (Python 3.9+)
-   **AI**: Google Gemini 2.5 Flash (Multimodal)
-   **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (Glassmorphism design)
-   **Concepts**: RAG (Retrieval Augmented Generation), Theory of Mind Simulation.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file with your API key:
   ```env
   GEMINI_API_KEY=your_key_here
   ```

3. **Run the App**:
   ```bash
   python3 -m uvicorn app:app --reload
   ```

4. **Navigate to**: `http://127.0.0.1:8000`

## 🧩 How it Works

The system uses keyword-based RAG to match incoming texts against a curated `social_patterns.json` database. These patterns are based on research into neurodivergent communication barriers, ensuring that advice is validating rather than demanding.
