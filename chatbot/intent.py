import re
import sys
from dataclasses import dataclass
from chatbot import config

def debug_print(msg: str):
    if config.DEBUG:
        print(f"[DEBUG:INTENT] {msg}", file=sys.stderr)

@dataclass
class IntentResult:
    should_retrieve: bool
    system_instruction: str
    mode_name: str

def detect_intent(query: str) -> IntentResult:
    """
    Analyze the user query to determine intent and operational mode.
    
    Returns:
        IntentResult containing:
        - should_retrieve: Whether to perform RAG search
        - system_instruction: Specific prompt instructions for this mode
        - mode_name: Human readable mode name (for debug)
    """
    debug_print(f"Starting intent detection for query: '{query}'")
    q_lower = query.lower().strip()
    debug_print(f"Normalized query: '{q_lower}'")
    
    # 1. CONVERSATION / CHIT-CHAT
    # Trigger: Greetings, phatic expressions
    debug_print("Testing CONVERSATION patterns...")
    conversation_triggers = [
        r"^hello", r"^hi\b", r"^hey\b", r"^greetings",
        r"^how are you", r"^what'?s up", r"^good (morning|afternoon|evening|night)",
        r"^thanks", r"^thank you"
    ]
    
    for pattern in conversation_triggers:
        if re.search(pattern, q_lower):
            debug_print(f"MATCH: Pattern '{pattern}' matched -> CONVERSATION mode")
            debug_print("Decision: should_retrieve=False (casual conversation)")
            return IntentResult(
                should_retrieve=False,
                system_instruction="The user is engaging in casual conversation. Be friendly, polite, and concise. Do not try to look up facts unless explicitly asked.",
                mode_name="CONVERSATION"
            )
    debug_print("No CONVERSATION pattern matched")

    # 2. TUTORIAL / INSTRUCTIONAL
    # Trigger: "How to", "Guide for", "Steps to"
    debug_print("Testing TUTORIAL patterns...")
    if re.search(r"^(how to|guide|tutorial|steps|way to)", q_lower):
        debug_print("MATCH: TUTORIAL pattern matched")
        debug_print("Decision: should_retrieve=True (tutorial mode)")
        # Note: We MIGHT want RAG for tutorials if the user asks "How to fix a flat tire",
        # but the user requested "Intent model layer needs to know when to not index".
        # For a pure "How to write a poem" or "How to be happy", RAG is noise.
        # But "How to install Linux" might need RAG.
        # Strategy: If it looks like a general skill, skip RAG. If it looks like a Specific Entity query, use RAG.
        # For simplicity in this prototype: Tutorials enable RAG (knowledge is helpful) but change the Style.
        
        # Strategy: If it looks like a general skill, skip RAG. If it looks like a Specific Entity query, use RAG.
        # For simplicity in this prototype: Tutorials enable RAG (knowledge is helpful) but change the Style.
        
        return IntentResult(
            should_retrieve=True,
            system_instruction=(
                "\nMODE: TUTORIAL\n"
                "OBJECTIVE: Provide a structured, easy-to-follow guide.\n"
                "LAYOUT RULES:\n"
                "1. TITLE: Start with a clear H1 title (e.g. '# How to...')\n"
                "2. OVERVIEW: Brief summary (1-2 sentences).\n"
                "3. STEPS: Use numbered lists for actions (1. **Do this**...).\n"
                "4. FORMATTING: Use **bold** for key terms/buttons. Use `code blocks` for commands.\n"
                "5. TONE: Helpful, instructional, and encouraging."
            ),
            mode_name="TUTORIAL"
        )
        
    debug_print("Testing DEBATE patterns...")
    # 3. DEBATE / OPINION (Experimental)
    if re.search(r"^(argue|debate|pros and cons|opinion on)", q_lower):
        debug_print("MATCH: DEBATE pattern matched")
        debug_print("Decision: should_retrieve=True (debate mode)")
        return IntentResult(
            should_retrieve=True,
            system_instruction="\nMODE: DEBATE\nPresent multiple viewpoints. Analyze pros and cons objectively. Use bullet points to contrast arguments.",
            mode_name="DEBATE"
        )

    # 4. FACTUAL / DEFAULT (The "Rest")
    # Trigger: "What is", "Who is", "When", "Define", or any specific query
    debug_print("No specific pattern matched -> FACTUAL (default) mode")
    debug_print("Decision: should_retrieve=True (factual mode)")
    return IntentResult(
        should_retrieve=True,
        system_instruction=(
            "\nMODE: FACTUAL REASONING\n"
            "OBJECTIVE: Provide a single, concise answer. If sources conflict, report the conflict neutrally.\n"
            "LAYOUT RULES:\n"
            "1. DIRECT ANSWER: Start directly with the answer.\n"
            "2. REPORT CONFLICTS: If sources disagree (e.g. Source A says X, Source B says Y), state BOTH. Do NOT try to resolve it.\n"
            "3. NO ARGUMENTATION: Do not debate with yourself (e.g. 'But wait...', 'However...'). just list the facts.\n"
            "4. NO REPETITION: State the facts once and STOP. Do NOT summarize at the end."
        ),
        mode_name="FACTUAL"
    )
