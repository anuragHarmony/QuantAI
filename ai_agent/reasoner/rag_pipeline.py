"""RAG (Retrieval Augmented Generation) pipeline for AI-powered Q&A."""

from typing import List, Optional, Dict
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from shared.models.knowledge import QueryResult, RetrievalContext
from shared.config.settings import settings


class RAGPipeline:
    """RAG pipeline that combines retrieval with LLM generation."""

    def __init__(
        self,
        knowledge_engine=None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.

        Args:
            knowledge_engine: KnowledgeEngine instance for retrieval
            openai_api_key: OpenAI API key (uses settings if not provided)
            model: LLM model to use (uses settings if not provided)
        """
        self.knowledge_engine = knowledge_engine

        if OpenAI is None:
            raise ImportError("OpenAI is required for RAG. Install with: pip install openai")

        api_key = openai_api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env")

        self.client = OpenAI(api_key=api_key)
        self.model = model or settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        logger.info(f"RAG Pipeline initialized with model: {self.model}")

    def ask(
        self,
        question: str,
        max_context_tokens: int = None,
        include_examples: bool = True,
        include_citations: bool = True
    ) -> Dict:
        """
        Ask a question and get an AI-generated answer with retrieved context.

        Args:
            question: User's question
            max_context_tokens: Maximum tokens for context (uses settings if not provided)
            include_examples: Whether to include examples in context
            include_citations: Whether to include source citations in response

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing question: {question}")

        # Retrieve relevant context
        max_tokens = max_context_tokens or settings.context_token_limit

        if self.knowledge_engine:
            context = self.knowledge_engine.get_context(
                query=question,
                max_tokens=max_tokens,
                include_examples=include_examples
            )
        else:
            # No knowledge engine - answer without context
            context = None

        # Generate answer
        answer_data = self._generate_answer(question, context, include_citations)

        logger.info(f"Generated answer with {len(answer_data.get('sources', []))} sources")

        return answer_data

    def _generate_answer(
        self,
        question: str,
        context: Optional[RetrievalContext],
        include_citations: bool
    ) -> Dict:
        """
        Generate answer using LLM with retrieved context.

        Args:
            question: User's question
            context: Retrieved context
            include_citations: Whether to include citations

        Returns:
            Dictionary with answer and metadata
        """
        # Build system prompt
        system_prompt = self._build_system_prompt(include_citations)

        # Build user prompt with context
        user_prompt = self._build_user_prompt(question, context)

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content

            # Extract sources
            sources = []
            if context and context.results:
                for result in context.results:
                    sources.append({
                        "book": result.chunk.source_book,
                        "chapter": result.chunk.source_chapter,
                        "type": result.chunk.chunk_type,
                        "relevance": result.relevance_score,
                        "preview": result.chunk.content[:200] + "..."
                    })

            return {
                "answer": answer,
                "question": question,
                "sources": sources,
                "confidence": context.confidence_score if context else 0.0,
                "num_sources": len(sources),
                "model": self.model,
                "context_tokens": context.total_tokens if context else 0
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _build_system_prompt(self, include_citations: bool) -> str:
        """Build the system prompt for the LLM."""
        base_prompt = """You are an expert quantitative finance analyst and trading strategist with deep knowledge of:
- Quantitative trading strategies
- Statistical analysis and modeling
- Risk management
- Backtesting methodologies
- Financial markets and instruments
- Trading algorithms

Your role is to provide accurate, detailed, and practical answers to questions about quantitative trading based on the provided context from research documents."""

        if include_citations:
            base_prompt += """

IMPORTANT: Always cite your sources when referencing specific information from the context. Use the format:
[Source: Book Name]

If you cannot find relevant information in the provided context, clearly state that and provide your best general knowledge answer while noting that it's not from the specific documents."""

        return base_prompt

    def _build_user_prompt(self, question: str, context: Optional[RetrievalContext]) -> str:
        """Build the user prompt with context."""
        if not context or not context.results:
            return f"""Question: {question}

Note: No specific context was found in the knowledge base. Please provide a general answer based on your knowledge of quantitative finance."""

        # Build context string from results
        context_str = "## Relevant Information from Knowledge Base:\n\n"

        for i, result in enumerate(context.results, 1):
            chunk = result.chunk
            context_str += f"### Source {i}: {chunk.source_book}\n"
            context_str += f"Type: {chunk.chunk_type} | Relevance: {result.relevance_score:.3f}\n"
            if chunk.source_chapter:
                context_str += f"Chapter: {chunk.source_chapter}\n"
            context_str += f"\nContent:\n{chunk.content}\n\n"
            context_str += "---\n\n"

        user_prompt = f"""{context_str}

## Question:
{question}

Please provide a comprehensive answer based on the information above. Be specific and reference the sources when possible."""

        return user_prompt

    def chat(
        self,
        messages: List[Dict[str, str]],
        include_retrieval: bool = True
    ) -> Dict:
        """
        Multi-turn conversation with RAG.

        Args:
            messages: List of message dicts with 'role' and 'content'
            include_retrieval: Whether to retrieve context for the last message

        Returns:
            Dictionary with answer and metadata
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        last_message = messages[-1]["content"]

        # Get context for last message if needed
        context = None
        if include_retrieval and self.knowledge_engine:
            context = self.knowledge_engine.get_context(
                query=last_message,
                max_tokens=settings.context_token_limit,
                include_examples=True
            )

        # Build system prompt
        system_prompt = self._build_system_prompt(include_citations=True)

        # Prepare messages for API
        api_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in messages[:-1]:
            api_messages.append(msg)

        # Add last message with context
        if context and context.results:
            enriched_message = self._build_user_prompt(last_message, context)
            api_messages.append({"role": "user", "content": enriched_message})
        else:
            api_messages.append(messages[-1])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content

            # Extract sources
            sources = []
            if context and context.results:
                for result in context.results:
                    sources.append({
                        "book": result.chunk.source_book,
                        "type": result.chunk.chunk_type,
                        "relevance": result.relevance_score
                    })

            return {
                "answer": answer,
                "sources": sources,
                "confidence": context.confidence_score if context else 0.0,
                "model": self.model
            }

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
