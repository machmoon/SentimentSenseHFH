from google import genai
from google.genai import types
import os
import json
import io
from dotenv import load_dotenv

from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS fix (so browser preflight OPTIONS works)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev/hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Load Knowledge Base (RAG)
try:
    with open("social_patterns.json", "r") as f:
        SOCIAL_PATTERNS = json.load(f)
except Exception:
    SOCIAL_PATTERNS = []

load_dotenv()  # only needed if using .env

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # Just print warning, don't crash, so user can fix .env while running
    print("❌ CAUTION: GEMINI_API_KEY not set. AI calls will fail.")
    client = None
else:
    client = genai.Client(api_key=api_key)


@app.post("/analyze")
async def analyze_multimodal(
    text: str = Form(None), 
    image: UploadFile = File(None),
    energy_level: str = Form("Medium")  # Spoon Theory Input
):
    """
    Analyze text and/or image input to determine social ambiguity and generate assistive responses.

    Parameters
    ----------
    text : str, optional
        The text message to analyze.
    image : UploadFile, optional
        A screenshot or image context to analyze.
    energy_level : str, optional
        The user's current energy/spoon level (Low, Medium, High). Default is "Medium".

    Returns
    -------
    dict
        A JSON object containing:
        - ambiguity_level (str): 'low', 'moderate', or 'high'.
        - confidence (float): 0.0 to 1.0.
        - sender_insight (str): Psychological analysis of the sender.
        - interpretations (list): Plausible interpretations.
        - response_paths (list): Drafted responses tailored to energy level.
    """
    if not client:
        return {"ambiguity_level": "Error", "interpretations": [{"summary": "API Key Missing"}]}

    # SIMPLE RAG LOGIC:
    # We use a keyword-based approach here (instead of Vector DB) because the 'social_patterns.json'
    # is small and curated. This ensures deterministic, high-quality advice for specific triggers
    # like "We need to talk" without the overhead/latency of embeddings.
    context_notes = []
    if text:
        text_lower = text.lower()
        for pattern in SOCIAL_PATTERNS:
            if any(k.lower() in text_lower for k in pattern['keywords']):
                 context_notes.append(f"- MATCHED RULE: {pattern['insight']}\n- ADVICE: {pattern['advice']}")
    
    rag_context = "\n".join(context_notes)
    
    # SYSTEM PROMPT DESIGN:
    # We define a "Social Forensics" persona to frame ambiguity as an external puzzle, not a user deficit.
    # The "Core Mission" (Validate, Relieve, Empower) is explicitly numbered to force the chain-of-thought
    # to prioritize emotional regulation before generating any practical advice.
    # We explicitly incorporate the "Double Empathy Problem" (Milton, 2012) – acknowledging that 
    # communication breakdown is a two-way street between different neurotypes.
    system_prompt = f"""
    You are an assistive communication system for neurodivergent (ADHD, Autistic, ND) users.
    
    CORE MISSION:
    1. **Validate**: Acknowledge that the user's confusion is valid. Neurotypical communication is often non-literal and indirect.
    2. **Relieve**: Reduce anxiety (RSD) by explaining the *unspoken rules* or logistical reasons behind the message.
    3. **Empower**: Offer "masking-optional" response paths. Help them express their needs clearly without forcing them to perform neurotypicality.

    NEURODIVERGENT CONTEXT:
    *   **ADHD**: Prioritize low-friction, immediate actions. Help bridge the "Shame Spiral" from late replies.
    *   **Autism**: Be literal, explicit, and logical. Explain metaphors and social "fluff" (phatic communication).
    *   **Double Empathy**: Avoid framing the user as "wrong". Frame the message as a "translation error" between two different communication styles.

    CURRENT USER STATE:
    **Energy Level (Spoons available)**: {energy_level}
    
    *   If **Low Spoons**: The user is in "Survival Mode". Drafts MUST be short, firm, and require zero emotional labor. Prioritize "Opt-out" scripts.
    *   If **High Spoons**: The user has energy to engage or mask if they choose. Drafts can be chatty and "socially lubricated".

    KNOWLEDGE BASE (RAG RULES):
    {rag_context}

    INSTRUCTIONS:
    1. Assess ambiguity (Low/Mod/High).
    2. Identify Interpretations (Literal vs. Social/Implicit).
    3. **Psychological Insight**: Explain the sender's likely mental state or social goal (e.g., they are in a rush, they are seeking reassurance, they are utilizing a social convention).
    4. Generate 3 Response Paths tailored to **{energy_level} Energy**:
       - Path 1: Low-Effort / Preservation (Minimal SPOONS).
       - Path 2: Medium-Effort / Standard.
       - Path 3: High-Effort / Investment (Maximize engagement).

    Return ONLY valid JSON in the following schema:
    {{
      "ambiguity_level": "low | moderate | high",
      "confidence": 0.0,
      "sender_insight": "A neuro-inclusive analysis of why the sender wrote this.",
      "interpretations": [
        {{
          "summary": "Validation-first description",
          "likely_goal": "literal info | social connection | reassurance | urgency | boundary"
        }}
      ],
      "response_paths": [
        {{
          "intent": "clarify | acknowledge | commit | set_boundary | re-entry",
          "description": "How this path protects your energy/autonomy",
          "draft_text": "text content"
        }}
      ]
    }}
    """
    
    # Build prompt content
    prompt_parts = [system_prompt]
    if text:
        prompt_parts.append(f"Message to analyze: {text}")

    if image:
        # Read image bytes
        img_bytes = await image.read()
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            prompt_parts.append(pil_img)
            # We add specific instruction for screenshots to flag "Time Gaps" and "Read Receipts"
            # as these are major sources of "Rejection Sensitive Dysphoria" (RSD) for our users.
            prompt_parts.append("Analyze this screenshot through the lens of social ambiguity.")
        except Exception as e:
             return {"ambiguity_level": "Error", "interpretations": [{"summary": str(e)}]}

    try:
        # Use the correct model name (2.5-flash is stable)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
        
    except Exception as e:
        return {"ambiguity_level": "High", "interpretations": [{"summary": f"API Error: {str(e)}"}]}


@app.post("/simulate")
async def simulate_reaction(
    original_text: str = Form(...),
    draft_text: str = Form(...)
):
    """
    Predict the reaction of the message sender to a proposed draft response.

    Parameters
    ----------
    original_text : str
        The original ambiguous message received by the user.
    draft_text : str
        The draft response the user is considering sending.

    Returns
    -------
    dict
        A JSON object containing:
        - likely_mood (str): The predicted emotional state of the sender (e.g., 'Relieved', 'Annoyed').
        - predicted_reply (str): A hypothetical text message the sender might write next.
    """
    if not client:
        return {"likely_mood": "API Key Missing", "predicted_reply": "Error"}

    # SIMULATION ENGINE:
    # Instead of just generating text, we ask the AI to "roleplay" the sender. 
    # This leverages the "Theory of Mind" capabilities of the LLM to predict emotional reactions,
    # allowing the user to "A/B Test" social interactions safely before committing.
    system_prompt = f"""
    You are a 'Social Simulation Engine'. 
    
    SCENARIO:
    Person A sent: "{original_text}"
    Person B (User) is considering replying: "{draft_text}"
    
    TASK:
    Predict Person A's likely reaction.
    1. Will they be relieved, annoyed, confused, or happy?
    2. Write their likely next text message.
    
    Return JSON: {{ "likely_mood": "Relieved | Annoyed | ...", "predicted_reply": "..." }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"likely_mood": "Error", "predicted_reply": str(e)}


@app.post("/breakdown")
async def breakdown_tasks(
    text: str = Form(...)
):
    """
    Break down a long or complex message into actionable tasks for users with ADHD.

    Parameters
    ----------
    text : str
        The long message or email to break down.

    Returns
    -------
    dict
        A JSON object containing a list of tasks.
    """
    if not client:
        return {"tasks": ["API Key Missing"]}

    system_prompt = """
    You are an ADHD Executive Function Assistant. 
    The user received a long message and is feeling overwhelmed. 
    
    TASK:
    1. Extract actionable items/tasks from the message.
    2. Simplify the language into clear "next steps".
    3. Identify if any item has a deadline.
    
    Return JSON: { "tasks": [ { "task": "...", "priority": "high|medium|low", "deadline": "..." } ] }
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, f"Message: {text}"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"tasks": [{"task": f"Error: {str(e)}", "priority": "low", "deadline": ""}]}
