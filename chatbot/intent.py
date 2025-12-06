import re
from dataclasses import dataclass

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
    q_lower = query.lower().strip()
    
    # 1. CONVERSATION / CHIT-CHAT
    # Trigger: Greetings, phatic expressions
    conversation_triggers = [
        r"^hello", r"^hi\b", r"^hey\b", r"^greetings",
        r"^how are you", r"^what'?s up", r"^good (morning|afternoon|evening|night)",
        r"^thanks", r"^thank you"
    ]
    
    for pattern in conversation_triggers:
        if re.search(pattern, q_lower):
            return IntentResult(
                should_retrieve=False,
                system_instruction="The user is engaging in casual conversation. Be friendly, polite, and concise. Do not try to look up facts unless explicitly asked.",
                mode_name="CONVERSATION"
            )

    # 2. TUTORIAL / INSTRUCTIONAL
    # Trigger: "How to", "Guide for", "Steps to"
    if re.search(r"^(how to|guide|tutorial|steps|way to)", q_lower):
        # Note: We MIGHT want RAG for tutorials if the user asks "How to fix a flat tire",
        # but the user requested "Intent model layer needs to know when to not index".
        # For a pure "How to write a poem" or "How to be happy", RAG is noise.
        # But "How to install Linux" might need RAG.
        # Strategy: If it looks like a general skill, skip RAG. If it looks like a Specific Entity query, use RAG.
        # For simplicity in this prototype: Tutorials enable RAG (knowledge is helpful) but change the Style.
        
        return IntentResult(
            should_retrieve=True,
            system_instruction="\nMODE: TUTORIAL\nStructure your answer as a clear, step-by-step tutorial. Use numbered lists, bold headers, and explain concepts simply like a teacher.",
            mode_name="TUTORIAL"
        )
        
    # 3. DEBATE / OPINION (Experimental)
    if re.search(r"^(argue|debate|pros and cons|opinion on)", q_lower):
        return IntentResult(
            should_retrieve=True,
            system_instruction="\nMODE: DEBATE\nPresent multiple viewpoints. Analyze pros and cons objectively. Use bullet points to contrast arguments.",
            mode_name="DEBATE"
        )

    # 4. FACTUAL / DEFAULT (The "Rest")
    # Trigger: "What is", "Who is", "When", "Define", or any specific query
    return IntentResult(
        should_retrieve=True,
        system_instruction="\nMODE: FACTUAL\nProvide a direct, accurate answer based strictly on the retrieved context.",
        mode_name="FACTUAL"
    )
