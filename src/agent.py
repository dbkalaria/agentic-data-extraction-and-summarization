from typing import List, Dict, Any

from core.config import settings
from core.prompts import NEWS_ANALYST_PROMPT
from core.logging_config import logger
from core.connections import (
    db,
    generative_model,
    embedding_model,
    index_endpoint
)

class NewsAnalystAgent:
    """
    An intelligent agent that analyzes a corpus of news articles to answer user queries.
    It uses a Retrieval-Augmented Generation (RAG) workflow on GCP.
    """

    def __init__(self, top_k: int = 3):
        """
        Initializes the agent.
        Args:
            top_k (int): The number of relevant documents to retrieve for context.
        """
        self.top_k = top_k
        logger.info(f"NewsAnalystAgent initialized. Will retrieve top {self.top_k} documents per query.")


    def _find_relevant_articles(self, query: str) -> List[str]:
        """
        Tool 1: Semantic Search.
        Finds the most relevant article IDs using Vertex AI Vector Search.
        """
        logger.info(f"[Tool 1] Performing semantic search for query: '{query}'")
        
        query_embedding = embedding_model.get_embeddings([query])[0].values
        
        response = index_endpoint.find_neighbors(
            queries=[query_embedding],
            deployed_index_id=settings.deployed_index_id,
            num_neighbors=self.top_k
        )
        
        article_ids = [neighbor.id for neighbor in response[0]]
        logger.info(f"[Tool 1] Found {len(article_ids)} relevant article IDs: {article_ids}")
        return article_ids

    def _get_article_context(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Tool 2: Context Retrieval.
        Fetches the content (summary, text chunk, entities) for the given article IDs from Firestore.
        """
        logger.info(f"[Tool 2] Fetching context for {len(article_ids)} articles from Firestore...")
        
        context_bundle = []
        docs_ref = db.collection(settings.firestore_collection)
        
        for doc_id in article_ids:
            doc = docs_ref.document(doc_id).get()
            if doc.exists:
                data = doc.to_dict()
                context_bundle.append({
                    "id": doc.id,
                    "summary": data.get("gemini_summary", "No summary available."),
                    "key_info": str(data.get("vertex_ai_extraction", "No key information extracted."))
                })
            else:
                logger.warning(f"[Tool 2] Article ID '{doc_id}' found in Vector Search but not in Firestore.")
                
        logger.info(f"[Tool 2] Successfully fetched context for {len(context_bundle)} articles.")
        return context_bundle

    def _synthesize_news_report(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Tool 3: Answer Synthesis.
        Uses the Gemini model to generate a synthesized answer based on the retrieved context.
        """
        logger.info("[Tool 3] Synthesizing final answer with Gemini Pro...")

        context_str = ""
        for i, doc in enumerate(context):
            context_str += f"--- Start of Source {i+1} ---\n"
            context_str += f"Source ID: {doc['id']}\n"
            context_str += f"Summary: {doc['summary']}\n"
            context_str += f"Key Information Extracted: {doc['key_info']}\n"
            context_str += f"--- End of Source {i+1} ---\n\n"

        prompt = NEWS_ANALYST_PROMPT.format(
            query=query,
            context_str=context_str
        )
        
        try:
            response = generative_model.generate_content(prompt)
            logger.info("[Tool 3] Successfully generated response from Gemini.")
            return response.text
        except Exception as e:
            logger.error(f"[Tool 3] Error calling Gemini API: {e}")
            return "There was an error while generating the final answer."

    def answer(self, query: str) -> str:
        """
        The main agentic loop that orchestrates the tools to answer a query.
        """
        logger.info(f"--- Starting new agent query: '{query}' ---")
        
        article_ids = self._find_relevant_articles(query)
        if not article_ids:
            return "I'm sorry, I could not find any relevant news articles for your query."

        context = self._get_article_context(article_ids)
        if not context:
            return "I found some article references, but I was unable to retrieve their content from the database."
            
        final_answer = self._synthesize_news_report(query, context)
        
        logger.info(f"--- Finished agent query ---")
        return final_answer


def main():
    """
    The main function to run the interactive agent.
    """
    agent = NewsAnalystAgent(top_k=10)
    
    print("\n--- News Analyst Agent ---")
    print("Ask me anything about the news articles in the database.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 26)
    
    while True:
        try:
            query = input("You: ")

            if not query.strip():
                continue

            if query.lower() in ['quit', 'exit']:
                print("Agent: Goodbye!")
                break
            
            print("\nAgent: Thinking...")
            
            final_answer = agent.answer(query)
            
            print(f"\nAgent:\n{final_answer}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nAgent: Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)


if __name__ == '__main__':
    main()