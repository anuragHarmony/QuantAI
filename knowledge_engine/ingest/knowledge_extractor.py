"""LLM-based knowledge extraction from document chunks."""

import json
from typing import List, Optional
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from shared.models.knowledge import KnowledgeChunk, ChunkType
from shared.config.settings import settings


class KnowledgeExtractor:
    """Extracts structured knowledge from text using LLMs."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the knowledge extractor.

        Args:
            api_key: OpenAI API key (uses settings if not provided)
        """
        if OpenAI is None:
            logger.warning("OpenAI not installed. Knowledge extraction will be limited.")
            self.client = None
        else:
            api_key = api_key or settings.openai_api_key
            if not api_key:
                logger.warning("No OpenAI API key provided. Using rule-based extraction only.")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)

    def extract_knowledge_chunks(
        self,
        text: str,
        source_book: str,
        source_chapter: Optional[str] = None,
        source_page: Optional[int] = None
    ) -> List[KnowledgeChunk]:
        """
        Extract knowledge chunks from text.

        Args:
            text: Text to extract knowledge from
            source_book: Name of source book
            source_chapter: Chapter name (optional)
            source_page: Page number (optional)

        Returns:
            List of KnowledgeChunk objects
        """
        if self.client:
            return self._extract_with_llm(text, source_book, source_chapter, source_page)
        else:
            return self._extract_rule_based(text, source_book, source_chapter, source_page)

    def _extract_with_llm(
        self,
        text: str,
        source_book: str,
        source_chapter: Optional[str],
        source_page: Optional[int]
    ) -> List[KnowledgeChunk]:
        """Extract knowledge using LLM."""
        logger.info("Extracting knowledge using LLM")

        prompt = f"""You are an expert quantitative finance analyst. Extract structured knowledge from the following text.

For each distinct concept, strategy, formula, or example, provide:
1. Type: concept, strategy, formula, example, or code
2. Hierarchy level: 1 (broad topic), 2 (subtopic), 3 (detailed), or 4 (example)
3. Content: The actual text content
4. Asset classes (if mentioned): equity, forex, crypto, commodities, etc.
5. Strategy types (if mentioned): trend_following, mean_reversion, arbitrage, etc.
6. Tags: relevant keywords

Text:
{text}

Return a JSON array where each object has:
{{
    "chunk_type": "concept|strategy|formula|example|code",
    "hierarchy_level": 1-4,
    "content": "the extracted content",
    "asset_class": ["equity", "forex"],
    "strategy_type": ["trend_following"],
    "tags": ["moving_average", "technical_analysis"]
}}

Only return the JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are a quantitative finance expert that extracts structured knowledge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content

            # Parse JSON response
            try:
                # Try to parse as direct JSON array
                parsed = json.loads(result_text)
                if isinstance(parsed, dict) and "chunks" in parsed:
                    extracted_data = parsed["chunks"]
                elif isinstance(parsed, list):
                    extracted_data = parsed
                else:
                    extracted_data = [parsed]
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {e}")
                return self._extract_rule_based(text, source_book, source_chapter, source_page)

            # Convert to KnowledgeChunk objects
            chunks = []
            for item in extracted_data:
                try:
                    chunk = KnowledgeChunk(
                        source_book=source_book,
                        source_chapter=source_chapter,
                        source_page=source_page,
                        chunk_type=ChunkType(item.get("chunk_type", "concept")),
                        hierarchy_level=item.get("hierarchy_level", 2),
                        content=item.get("content", ""),
                        asset_class=item.get("asset_class", []),
                        strategy_type=item.get("strategy_type", []),
                        tags=item.get("tags", [])
                    )
                    chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Error creating chunk: {e}")
                    continue

            logger.info(f"Extracted {len(chunks)} knowledge chunks using LLM")
            return chunks

        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return self._extract_rule_based(text, source_book, source_chapter, source_page)

    def _extract_rule_based(
        self,
        text: str,
        source_book: str,
        source_chapter: Optional[str],
        source_page: Optional[int]
    ) -> List[KnowledgeChunk]:
        """
        Extract knowledge using simple rule-based approach.
        Fallback when LLM is not available.
        """
        logger.info("Extracting knowledge using rule-based approach")

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        for para in paragraphs:
            # Skip very short paragraphs
            if len(para) < 50:
                continue

            # Determine chunk type based on keywords
            chunk_type = self._detect_chunk_type(para)
            hierarchy_level = self._detect_hierarchy_level(para)
            tags = self._extract_tags(para)

            chunk = KnowledgeChunk(
                source_book=source_book,
                source_chapter=source_chapter,
                source_page=source_page,
                chunk_type=chunk_type,
                hierarchy_level=hierarchy_level,
                content=para,
                tags=tags
            )
            chunks.append(chunk)

        logger.info(f"Extracted {len(chunks)} knowledge chunks using rules")
        return chunks

    def _detect_chunk_type(self, text: str) -> ChunkType:
        """Detect the type of knowledge chunk based on content."""
        text_lower = text.lower()

        # Check for code
        if 'def ' in text or 'import ' in text or 'class ' in text:
            return ChunkType.CODE

        # Check for formulas
        if any(symbol in text for symbol in ['=', '∑', '∫', 'β', 'σ', '√']):
            return ChunkType.FORMULA

        # Check for examples
        if any(word in text_lower for word in ['example', 'for instance', 'consider', 'suppose']):
            return ChunkType.EXAMPLE

        # Check for strategies
        if any(word in text_lower for word in ['strategy', 'trading', 'signal', 'backtest']):
            return ChunkType.STRATEGY

        # Default to concept
        return ChunkType.CONCEPT

    def _detect_hierarchy_level(self, text: str) -> int:
        """Detect hierarchy level based on content characteristics."""
        # Short text is usually higher level
        if len(text) < 200:
            return 1

        # Medium length is mid-level
        if len(text) < 500:
            return 2

        # Long detailed text
        if len(text) < 1000:
            return 3

        # Very long is usually examples
        return 4

    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text."""
        tags = []
        text_lower = text.lower()

        # Common quant finance terms
        keywords = [
            'moving_average', 'rsi', 'macd', 'bollinger',
            'momentum', 'trend', 'mean_reversion', 'volatility',
            'arbitrage', 'pairs_trading', 'portfolio', 'risk',
            'sharpe', 'returns', 'drawdown', 'backtest',
            'equity', 'forex', 'crypto', 'options'
        ]

        for keyword in keywords:
            if keyword.replace('_', ' ') in text_lower or keyword in text_lower:
                tags.append(keyword)

        return tags[:10]  # Limit to 10 tags
