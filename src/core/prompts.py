"""
prompts.py
----------

This module centralizes all the prompts used in the project.
Storing prompts in a dedicated file makes them easier to manage, test, and version.

Each prompt is a string variable that can be imported and used in other modules.
"""

# --- Agent Prompts ---

NEWS_ANALYST_PROMPT = """
        You are an expert News Analyst. Your task is to provide a clear, factual, and synthesized answer to the user's question.
        Base your answer *exclusively* on the provided news article sources.
        Do not use any external knowledge.
        When you use information from a source, cite it at the end of the sentence using its ID (e.g., [document-id-123]).
        If the sources do not contain enough information to answer the question, state that clearly.

        **User's Question:**
        {query}

        **Provided Sources:**
        {context_str}

        **Synthesized News Report:**
"""


# --- Information Extraction Prompts ---

ENTITY_EXTRACTION_PROMPT = """
        **Instruction:**
        From the news article provided below, extract the key information.

        **JSON Schema to follow:**
        - "main_event_or_topic": A concise string describing the central event or subject of the article.
        - "key_people": A list of key individuals mentioned.
        - "key_organizations": A list of key organizations, companies, or government bodies.
        - "key_locations": A list of key locations mentioned.
        - "dates_and_times": A list of specific dates or timeframes mentioned.
        - "quantitative_information": A list of important numbers, figures, or statistics (e.g., money, percentages, counts).
        - "outcome_or_impact": A brief string describing the result, consequence, or impact of the main event.

        **Article:**
        {text_content}
"""


# --- Summarization Prompts ---

GEMINI_SUMMARIZATION_PROMPT = """**Instruction:**
Your task is to write a single, high-quality sentence that summarizes the main point of the news article provided. The sentence should be written as if it were the first sentence of the article itself.

**Constraints:**
- The summary MUST be a single, complete sentence.
- It must be written in the third person.
- It must be highly factual and grounded in the provided text.
- Do not use introductory phrases like "This article discusses..." or "The document is about...".

**Article:**
{text_content}

**One-Sentence Summary:**
"""
