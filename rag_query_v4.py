#!/usr/bin/env python3
import os
import json
import re
import time
import hashlib
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import lru_cache
import logging

import yaml
from pydantic import BaseModel, validator
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from transformers import pipeline

from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---------------------------------------------------------------------------
# 🔧 Structured Logging Setup
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory()
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# 🛠️ Environment
# ---------------------------------------------------------------------------

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "Documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# ---------------------------------------------------------------------------
# 📐 Configuration Models
# ---------------------------------------------------------------------------

class PatternConfig(BaseModel):
    """Configuration for document classification patterns."""
    name: str
    description: str
    priority: int
    patterns: Dict[str, List[str]]
    boost_keywords: List[str]
    temporal_importance: bool

    @validator("patterns")
    def validate_patterns(cls, v):
        if not v:
            raise ValueError("Patterns must not be empty")
        return v

class LanguagePatterns(BaseModel):
    """Language-specific patterns for classification and intent detection."""
    language: str
    patterns: Dict[str, PatternConfig]

class Config(BaseModel):
    """Top-level configuration for document types and languages."""
    document_types: List[PatternConfig]
    languages: Dict[str, LanguagePatterns]

# ---------------------------------------------------------------------------
# 📄 Document Classifier (EN/ES)
# ---------------------------------------------------------------------------

class EnhancedDocumentClassifier:
    """Advanced document classifier with bilingual support and caching."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config file."""
        self.load_config(config_path)
        self._compile_patterns()

    def load_config(self, config_path: Optional[str]) -> None:
        """Load configuration from YAML or use default."""
        default_config = {
            "document_types": [
                {
                    "name": "personal_ownership",
                    "description": "Personal ownership documents and records / Documentos de propiedad personal",
                    "priority": 10,
                    "patterns": {
                        "ownership_indicators": [
                            # English
                            r"\bowned\s+by\b", r"\bbelongs\s+to\b", r"\bmy\s+\w+", r"\bpurchased\s+by\b", r"\bacquired\s+by\b",
                            # Spanish
                            r"\bpropiedad\s+de\b", r"\bpertenece\s+a\b", r"\bmi\s+\w+", r"\bm[ií]o\b", r"\bcomprado\s+por\b", r"\badquirido\s+por\b"
                        ],
                        "document_types": [
                            # English
                            r"\breceipt\b", r"\binvoice\b", r"\bwarranty\b", r"\bcertificate\b", r"\bdeed\b", r"\btitle\b",
                            # Spanish
                            r"\brecibo\b", r"\bfactura\b", r"\bgarant(?:[íi])a\b", r"\bcertificado\b", r"\bescritura\b", r"\bt[íi]tulo\b"
                        ],
                        "possession_verbs": [
                            # English
                            r"\bhave\s+\w+", r"\bown\s+\w+", r"\bpossess\s+\w+",
                            # Spanish
                            r"\btengo\s+\w+", r"\bposeo\s+\w+", r"\bdispongo\s+de\s+\w+"
                        ]
                    },
                    "boost_keywords": [
                        # English
                        "receipt", "purchased", "warranty", "certificate", "own", "mine",
                        # Spanish
                        "recibo", "comprado", "garantía", "certificado", "poseo", "mío"
                    ],
                    "temporal_importance": True
                },
                # ... (other document types as in original)
            ],
            "languages": {
                "en": {"language": "en", "patterns": {}},
                "es": {"language": "es", "patterns": {}}
            }
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self.config = Config(**config_data)
        else:
            self.config = Config(**default_config)
        self.patterns = self.config.document_types

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self.compiled_patterns = {}
        for dtype in self.patterns:
            name = dtype.name
            self.compiled_patterns[name] = {}
            for group, patterns in dtype.patterns.items():
                self.compiled_patterns[name][group] = [re.compile(p, re.IGNORECASE) for p in patterns]

    @lru_cache(maxsize=1000)
    def _classify_document(self, content_hash: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document with caching based on content hash."""
        content_lower = content.lower()
        path = metadata.get("path", "").lower()
        title = metadata.get("title", "").lower()
        tags = metadata.get("tags", [])
        last_modified = metadata.get("last_modified", "")

        classifications = []
        for dtype in self.patterns:
            name = dtype.name
            score = 0
            matched = []
            for group, regexes in self.compiled_patterns[name].items():
                gm = sum(1 for rgx in regexes if rgx.search(content))
                weight = 3 if group == "ownership_indicators" else 2 if group == "document_types" else 1
                score += gm * weight
                matched.extend([f"{group}:{rgx.pattern[:30]}" for rgx in regexes if rgx.search(content)])
            for kw in dtype.boost_keywords:
                if kw in content_lower:
                    score += 1
                if kw in title:
                    score += 2
                if kw in tags:
                    score += 1.5
            if any(h in path for h in ["recibo", "receipt", "manual", "guía", "guide", "doc"]):
                score += 1
            if dtype.temporal_importance and last_modified:
                try:
                    mod_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    days_old = (datetime.now() - mod_date.replace(tzinfo=None)).days
                    score += 2 if days_old < 30 else 1 if days_old < 90 else 0
                except ValueError:
                    pass
            confidence = min(score * 0.15, 1.0)
            classifications.append({
                "type": name,
                "description": dtype.description,
                "confidence": confidence,
                "score": score,
                "priority": dtype.priority,
                "matched_patterns": matched[:5]
            })
        classifications.sort(key=lambda x: x["confidence"] * x["priority"], reverse=True)
        primary = classifications[0] if classifications else {"type": "unknown", "confidence": 0.0}
        return {
            "primary_classification": primary,
            "all_classifications": classifications,
            "metadata_used": {
                "has_tags": bool(tags),
                "has_path": bool(path),
                "has_timestamp": bool(last_modified)
            }
        }

    def classify_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document with content hashing."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return self._classify_document(content_hash, content, metadata)

# ---------------------------------------------------------------------------
# 📐 Data Classes
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Represents detected query intent."""
    intent_type: str
    confidence: float
    keywords: List[str]
    temporal_context: Optional[str] = None
    entities: List[str] = field(default_factory=list)

@dataclass
class ConversationContext:
    """Maintains conversation state and memory."""
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    current_topic: Optional[str] = None
    mentioned_entities: Set[str] = field(default_factory=set)
    last_query_intent: Optional[QueryIntent] = None

    def add_exchange(self, query: str, response: str, intent: QueryIntent) -> None:
        """Add query-response exchange to history."""
        self.history.append({
            'query': query,
            'response': response,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        self.last_query_intent = intent
        entities = self._extract_entities(query)
        self.mentioned_entities.update(entities)
        if intent.intent_type in ['search', 'ownership']:
            self.current_topic = ' '.join(intent.keywords[:3])

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities using regex."""
        entities = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', text)
        entities.extend(re.findall(r'"([^"]*)"', text))
        entities.extend(re.findall(r"'([^']*)'", text))
        return entities

    def get_context_summary(self) -> str:
        """Summarize recent conversation history."""
        if not self.history:
            return ""
        recent = list(self.history)[-3:]
        return "\n".join(f"Q: {ex['query'][:100]}\nA: {ex['response'][:100]}…" for ex in recent)

# ---------------------------------------------------------------------------
# 🔍 Intent Detector (EN/ES)
# ---------------------------------------------------------------------------

class QueryIntentDetector:
    """Detects query intent using zero-shot classification and regex fallback."""

    def __init__(self):
        """Initialize intent classifier and regex patterns."""
        try:
            self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            logger.warning(f"Failed to load intent classifier: {e}, using regex fallback")
            self.intent_classifier = None
        self.intent_patterns = {
            "ownership": [
                r"\bdo\s+i\s+(have|own|possess)\b", r"\bwhat\s+do\s+i\s+(have|own|possess)\b", r"\bmy\s+\w+", r"\bmine\b", r"\bshow\s+me\s+what\s+i\b",
                r"\b(tengo|poseo|dispongo)\b", r"\bqu[ée]\s+(tengo|poseo)\b", r"\bmi\s+\w+", r"\bm(?:í|i)o\b"
            ],
            "search": [
                r"\bfind\b", r"\bsearch\b", r"\blook\s+for\b", r"\bshow\s+me\b", r"\bwhere\s+is\b",
                r"\bbuscar\b", r"\bencuentra?\b", r"\bd[óo]nde\s+est[áa]\b", r"\bmuestra?me\b", r"\benséñame\b"
            ],
            "comparison": [
                r"\bcompare\b", r"\bdifference\b", r"\bvs\b", r"\bbetter\b", r"\bworse\b", r"\bwhich\s+is\b",
                r"\bcompara?r?\b", r"\bdiferencia\b", r"\bmejor\b", r"\bpeor\b", r"\bcu[aá]l\s+es\b"
            ],
            "temporal": [
                r"\bwhen\b", r"\brecent\b", r"\blast\s+\w+\b", r"\btoday\b", r"\byesterday\b", r"\bthis\s+\w+\b", r"\bsince\b", r"\buntil\b", r"\bbefore\b", r"\bafter\b",
                r"\bcu[aá]ndo\b", r"\breciente\b", r"\b(la|el)\s+\w+\s+pasad[ao]\b", r"\bhoy\b", r"\bayer\b", r"\besta\s+\w+\b", r"\bdesde\b", r"\bhasta\b", r"\bantes\b", r"\bdespu[eé]s\b"
            ],
            "procedural": [
                r"\bhow\s+to\b", r"\bsteps?\b", r"\bprocess\b", r"\bprocedure\b", r"\bmethod\b", r"\bway\s+to\b",
                r"\bc[óo]mo\b", r"\bpasos?\b", r"\bproceso\b", r"\bprocedimiento\b", r"\bm[ée]todo\b", r"\bforma\s+de\b"
            ]
        }
        self.compiled_patterns = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in self.intent_patterns.items()}

    def detect_intent(self, query: str) -> QueryIntent:
        """Detect intent using zero-shot classification or regex fallback."""
        if self.intent_classifier:
            try:
                candidate_labels = ["ownership", "search", "comparison", "temporal", "procedural"]
                result = self.intent_classifier(query, candidate_labels, multi_label=True)
                intents = [
                    {"intent_type": label, "confidence": score}
                    for label, score in zip(result["labels"], result["scores"])
                    if score > 0.5
                ]
                primary = intents[0] if intents else {"intent_type": "search", "confidence": 0.3}
                intent_type, confidence = primary["intent_type"], primary["confidence"]
            except Exception as e:
                logger.warning(f"Model-based intent detection failed: {e}, using regex")
                intent_type, confidence = self._regex_based_intent(query)
        else:
            intent_type, confidence = self._regex_based_intent(query)

        temporal_context = self._extract_temporal_context(query.lower())
        keywords = self._extract_keywords(query)
        entities = self._extract_entities(query)
        return QueryIntent(intent_type, confidence, keywords, temporal_context, entities)

    def _regex_based_intent(self, query: str) -> Tuple[str, float]:
        """Fallback regex-based intent detection."""
        q_lower = query.lower()
        scores = defaultdict(float)
        for intent, pats in self.compiled_patterns.items():
            for pat in pats:
                if pat.search(q_lower):
                    scores[intent] += 1
        if scores:
            primary = max(scores, key=scores.get)
            return primary, min(scores[primary] * 0.3, 1.0)
        return "search", 0.3

    def _extract_temporal_context(self, query: str) -> Optional[str]:
        """Extract temporal context from query."""
        temporal_patterns = [
            r"\b(today|hoy|yesterday|ayer|last\s+week|la\s+semana\s+pasada|last\s+month|el\s+mes\s+pasado|this\s+year|este\s+a[ñn]o)\b",
            r"\b(before|after|antes|despu[eé]s|since|desde|until|hasta)\s+(\w+)\b",
            r"\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})\b"
        ]
        for p in temporal_patterns:
            if m := re.search(p, query, re.IGNORECASE):
                return m.group(0)
        return None

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        kws = re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]{3,}\b", query)
        stop = {'the', 'and', 'for', 'are', 'have', 'with', 'this', 'that',
                'los', 'las', 'unos', 'unas', 'para', 'con', 'este', 'esta', 'ese', 'esa', 'son', 'tengo'}
        return [k for k in kws if k.lower() not in stop][:10]

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query."""
        entities = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', query)
        entities.extend(re.findall(r'"([^"]*)"', query))
        entities.extend(re.findall(r"'([^']*)'", query))
        entities.extend(re.findall(r'\b\d+[A-Za-z]+\b', query))
        return entities

# ---------------------------------------------------------------------------
# 📚 Enhanced Retriever
# ---------------------------------------------------------------------------

class EnhancedRetriever:
    """Retrieval layer with bilingual patterns and context awareness."""

    def __init__(self, vectorstore: WeaviateVectorStore, client: weaviate.WeaviateClient, index_name: str, embedder: HuggingFaceEmbeddings):
        """Initialize retriever with vectorstore and embedder."""
        self.vectorstore = vectorstore
        self.client = client
        self.collection = client.collections.get(index_name)
        self.embedder = embedder

    def retrieve_with_strategy(self, query: str, intent: QueryIntent, ctx: ConversationContext, k: int = 12) -> List[Document]:
        """Retrieve documents based on query intent and context."""
        base = self._vector_search(query, k=k*2)
        if intent.intent_type == "ownership":
            docs = self._filter_ownership_docs(base)
        elif intent.intent_type == "temporal":
            docs = self._filter_temporal_docs(base, intent)
        elif intent.intent_type == "procedural":
            docs = self._boost_procedural_docs(base)
        else:
            docs = base
        docs = self._apply_conversation_context(docs, ctx)
        docs = self._rerank_documents(docs, query, intent)
        return docs[:k]

    def _vector_search(self, query: str, k: int) -> List[Document]:
        """Perform vector-based similarity search."""
        return self.vectorstore.similarity_search(query, k=k)

    def _filter_ownership_docs(self, docs: List[Document]) -> List[Document]:
        """Filter documents with ownership indicators."""
        indicators = [
            "purchased", "bought", "own", "mine", "receipt", "invoice", "warranty", "certificate", "my", "belongs to",
            "comprado", "adquirido", "poseo", "tengo", "mío", "recibo", "factura", "garantía", "certificado", "mi", "pertenece a"
        ]
        selected = []
        for d in docs:
            lower = d.page_content.lower()
            score = sum(1 for ind in indicators if ind in lower)
            if d.metadata.get("tags"):
                score += sum(1 for tag in d.metadata["tags"] if any(ind in tag.lower() for ind in indicators)) * 2
            if score:
                d.metadata["ownership_score"] = score
                selected.append(d)
        return selected or docs

    def _filter_temporal_docs(self, docs: List[Document], intent: QueryIntent) -> List[Document]:
        """Filter documents by temporal relevance."""
        if not intent.temporal_context:
            return docs
        ctx = intent.temporal_context.lower()
        now = datetime.now()
        if any(k in ctx for k in ["today", "hoy"]):
            date_target = now.date()
        elif any(k in ctx for k in ["yesterday", "ayer"]):
            date_target = (now - timedelta(days=1)).date()
        elif any(k in ctx for k in ["last week", "la semana pasada"]):
            date_target = (now - timedelta(weeks=1)).date()
        elif any(k in ctx for k in ["last month", "el mes pasado"]):
            date_target = (now - timedelta(days=30)).date()
        else:
            return docs
        out = []
        for d in docs:
            lm = d.metadata.get("last_modified")
            if not lm:
                continue
            try:
                d_date = datetime.fromisoformat(lm.replace('Z', '+00:00')).date()
                diff = abs((d_date - date_target).days)
                if diff <= 7:
                    d.metadata["temporal_relevance"] = 1 - diff/7
                    out.append(d)
            except ValueError:
                continue
        return out if out else docs

    def _boost_procedural_docs(self, docs: List[Document]) -> List[Document]:
        """Boost documents with procedural content."""
        indicators = [
            "step", "procedure", "method", "process", "how to", "instructions", "guide", "manual", "tutorial",
            "paso", "procedimiento", "método", "proceso", "cómo", "instrucciones", "guía", "manual", "tutorial"
        ]
        for d in docs:
            lower = d.page_content.lower()
            score = sum(1 for ind in indicators if ind in lower)
            if re.search(r'\d+\.\s+', d.page_content):
                score += 2
            if re.search(r'\b(first|then|next|finally|lastly|primero|luego|después|finalmente)\b', lower):
                score += 1
            d.metadata["procedural_score"] = score
        return sorted(docs, key=lambda x: x.metadata.get("procedural_score", 0), reverse=True)

    def _apply_conversation_context(self, docs: List[Document], ctx: ConversationContext) -> List[Document]:
        """Boost documents based on conversation context."""
        if not ctx.mentioned_entities:
            return docs
        for d in docs:
            cscore = sum(1 for ent in ctx.mentioned_entities if ent.lower() in d.page_content.lower())
            if ctx.current_topic:
                cscore += sum(0.5 for w in ctx.current_topic.split() if w.lower() in d.page_content.lower())
            d.metadata["context_score"] = cscore
        return docs

    def _rerank_documents(self, docs: List[Document], query: str, intent: QueryIntent) -> List[Document]:
        """Rerank documents based on intent and relevance."""
        q_words = set(query.lower().split())
        for d in docs:
            score = 1.0
            if intent.intent_type == "ownership":
                score += d.metadata.get("ownership_score", 0) * 0.5
            elif intent.intent_type == "procedural":
                score += d.metadata.get("procedural_score", 0) * 0.3
            score += d.metadata.get("temporal_relevance", 0) * 0.4
            score += d.metadata.get("context_score", 0) * 0.2
            d_words = set(d.page_content.lower().split())
            overlap = len(q_words & d_words) / (len(q_words) or 1)
            score += overlap * 0.3
            d.metadata["final_relevance_score"] = score
        return sorted(docs, key=lambda x: x.metadata.get("final_relevance_score", 0), reverse=True)

# ---------------------------------------------------------------------------
# 🚀 Enhanced RAG System
# ---------------------------------------------------------------------------

class EnhancedRAG:
    """Enhanced Retrieval-Augmented Generation system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize RAG system with components."""
        self.client = self._make_weaviate_client(WEAVIATE_URL)
        if not self.client.is_ready():
            raise RuntimeError("Weaviate instance is not ready")
        
        self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=WEAVIATE_INDEX,
            text_key="text",
            embedding=self.embedder,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.llm = OllamaLLM(model=OLLAMA_MODEL)
        self.classifier = EnhancedDocumentClassifier(config_path)
        self.intent_detector = QueryIntentDetector()
        self.retriever = EnhancedRetriever(self.vectorstore, self.client, WEAVIATE_INDEX, self.embedder)
        self.conversation_context = ConversationContext()

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "intent", "conversation_history"],
            template="""You are an advanced AI assistant helping someone search their personal knowledge base with deep understanding of context and intent.

QUERY INTENT: {intent}
CONVERSATION CONTEXT: {conversation_history}

When answering:
1. Consider the detected intent and adapt your response style accordingly
2. For ownership queries ("do I have X?"), focus on definitive evidence and clear ownership statements
3. For procedural queries ("how to X?"), provide step-by-step guidance and prioritize instructional content
4. For temporal queries, emphasize recent or time-relevant information
5. Use conversation context to maintain coherent dialogue and avoid repetition
6. Cite specific evidence and document sources when making claims
7. Distinguish between what the user owns/has vs. general information
8. If confidence is low, acknowledge uncertainty and suggest alternatives

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Answer:"""
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 1}),
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_weaviate_client(self, url: str) -> weaviate.WeaviateClient:
        """Create Weaviate client with retry logic."""
        try:
            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 8080)
            if host in {"localhost", "127.0.0.1"}:
                return weaviate.connect_to_local(
                    host=host, port=port, grpc_port=50051,
                    additional_config=AdditionalConfig(timeout=Timeout(init=5))
                )
            return weaviate.connect_to_custom(
                http_host=host, http_port=port, http_secure=(parsed.scheme == "https"),
                grpc_host=host, grpc_port=50051, grpc_secure=(parsed.scheme == "https"),
                additional_config=AdditionalConfig(timeout=Timeout(init=5))
            )
        except weaviate.exceptions.WeaviateStartUpError as e:
            logger.error(f"Weaviate connection failed: {e}")
            raise

    def ingest_document(self, content: str, metadata: Dict[str, Any]) -> None:
        """Ingest document with chunking."""
        chunks = self.text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            classification = self.classifier.classify_document(chunk, chunk_metadata)
            chunk_metadata["classification"] = classification["primary_classification"]
            self.vectorstore.add_texts(
                texts=[chunk],
                metadatas=[chunk_metadata]
            )

    def query(self, query: str, include_analysis: Optional[bool] = None) -> str:
        """Process query with intent detection and context awareness."""
        start_time = time.time()
        logger.info("Processing query", query=query[:50], include_analysis=include_analysis)
        try:
            intent = self.intent_detector.detect_intent(query)
            logger.info(f"Detected intent: {intent.intent_type}", confidence=intent.confidence)

            if include_analysis is None:
                include_analysis = intent.intent_type in ["ownership", "comparison"] or intent.confidence > 0.7

            relevant_docs = self.retriever.retrieve_with_strategy(query, intent, self.conversation_context, k=12)
            context = self._prepare_context(relevant_docs)
            conversation_history = self.conversation_context.get_context_summary()

            intent_str = f"Type: {intent.intent_type}, Confidence: {intent.confidence:.2f}, Keywords: {', '.join(intent.keywords[:5])}"
            if intent.temporal_context:
                intent_str += f", Temporal: {intent.temporal_context}"

            response = self.llm.invoke(
                self.prompt_template.format(
                    context=context,
                    question=query,
                    intent=intent_str,
                    conversation_history=conversation_history
                )
            )

            if include_analysis:
                response += f"\n\n{self._generate_analysis(relevant_docs, intent, query)}"

            self.conversation_context.add_exchange(query, response, intent)
            logger.info("Query completed", duration=time.time() - start_time, num_docs=len(relevant_docs))
            return response.strip()

        except Exception as e:
            logger.error(f"Query processing error: {e}", query=query)
            return f"⚠️ Error processing query: {e}"

    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(docs):
            title = doc.metadata.get("title", f"Document {i+1}")
            path = doc.metadata.get("path", "")
            tags = doc.metadata.get("tags", [])
            context_part = f"[Document {i+1}: {title}"
            if path:
                context_part += f" | Path: {path}"
            if tags:
                context_part += f" | Tags: {', '.join(tags)}"
            if "final_relevance_score" in doc.metadata:
                context_part += f" | Relevance: {doc.metadata['final_relevance_score']:.2f}"
            context_part += f"]\n{doc.page_content}\n"
            context_parts.append(context_part)
        return "\n---\n".join(context_parts)

    def _generate_analysis(self, docs: List[Document], intent: QueryIntent, query: str) -> str:
        """Generate analysis of retrieved documents."""
        analysis_parts = ["📊 Document Analysis:"]
        analysis_parts.append(f"• Found {len(docs)} relevant documents")
        analysis_parts.append(f"• Query intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")

        doc_types = defaultdict(int)
        high_confidence_docs = 0
        for doc in docs:
            if "classification" in doc.metadata:
                classification = doc.metadata["classification"]
                doc_types[classification.get("type", "unknown")] += 1
                if classification.get("confidence", 0) > 0.7:
                    high_confidence_docs += 1
            else:
                doc_types["unclassified"] += 1

        if doc_types:
            analysis_parts.append("• Document types found:")
            for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
                analysis_parts.append(f"  - {doc_type.replace('_', ' ').title()}: {count}")

        if high_confidence_docs:
            analysis_parts.append(f"• High-confidence matches: {high_confidence_docs}")

        if intent.intent_type == "ownership":
            ownership_docs = [doc for doc in docs if doc.metadata.get("ownership_score", 0) > 0]
            if ownership_docs:
                analysis_parts.append(f"• Documents with ownership indicators: {len(ownership_docs)}")
                for doc in sorted(ownership_docs, key=lambda d: d.metadata.get("ownership_score", 0), reverse=True)[:3]:
                    title = doc.metadata.get("title", "Untitled")
                    score = doc.metadata.get("ownership_score", 0)
                    analysis_parts.append(f"  - {title} (ownership score: {score})")

        elif intent.intent_type == "procedural":
            procedural_docs = [doc for doc in docs if doc.metadata.get("procedural_score", 0) > 0]
            if procedural_docs:
                analysis_parts.append(f"• Procedural/how-to documents: {len(procedural_docs)}")

        elif intent.intent_type == "temporal" and intent.temporal_context:
            temporal_docs = [doc for doc in docs if doc.metadata.get("temporal_relevance", 0) > 0]
            if temporal_docs:
                analysis_parts.append(f"• Time-relevant documents: {len(temporal_docs)}")
                analysis_parts.append(f"• Temporal context: {intent.temporal_context}")

        if docs:
            analysis_parts.append("• Top relevant documents:")
            for i, doc in enumerate(docs[:3]):
                title = doc.metadata.get("title", f"Document {i+1}")
                relevance = doc.metadata.get("final_relevance_score", 0)
                analysis_parts.append(f"  {i+1}. {title} (relevance: {relevance:.2f})")

        return "\n".join(analysis_parts)

    def get_conversation_summary(self) -> str:
        """Summarize conversation history."""
        if not self.conversation_context.history:
            return "No conversation history."
        summary_parts = ["🔄 Conversation Summary:"]
        summary_parts.append(f"• Total exchanges: {len(self.conversation_context.history)}")
        if self.conversation_context.current_topic:
            summary_parts.append(f"• Current topic: {self.conversation_context.current_topic}")
        if self.conversation_context.mentioned_entities:
            entities = list(self.conversation_context.mentioned_entities)
            summary_parts.append(f"• Mentioned entities: {', '.join(entities[:5])}")
        summary_parts.append("• Recent queries:")
        for exchange in list(self.conversation_context.history)[-3:]:
            intent = exchange.get("intent", QueryIntent("unknown", 0.0, []))
            summary_parts.append(f"  - {exchange['query'][:50]}... (intent: {intent.intent_type})")
        return "\n".join(summary_parts)

    def reset_conversation(self) -> None:
        """Reset conversation context."""
        self.conversation_context = ConversationContext()
        logger.info("Conversation context reset")

    def debug_query_processing(self, query: str) -> Dict[str, Any]:
        """Debug query processing pipeline."""
        debug_info = {}
        intent = self.intent_detector.detect_intent(query)
        debug_info["intent"] = {
            "type": intent.intent_type,
            "confidence": intent.confidence,
            "keywords": intent.keywords,
            "temporal_context": intent.temporal_context,
            "entities": intent.entities
        }
        docs = self.retriever.retrieve_with_strategy(query, intent, self.conversation_context, k=5)
        debug_info["retrieval"] = {
            "num_docs": len(docs),
            "documents": [
                {
                    "title": doc.metadata.get("title", "Untitled"),
                    "path": doc.metadata.get("path", ""),
                    "relevance_score": doc.metadata.get("final_relevance_score", 0),
                    "ownership_score": doc.metadata.get("ownership_score", 0),
                    "procedural_score": doc.metadata.get("procedural_score", 0),
                    "context_score": doc.metadata.get("context_score", 0),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "classification": doc.metadata.get("classification", {})
                } for doc in docs
            ]
        }
        debug_info["conversation_context"] = {
            "history_length": len(self.conversation_context.history),
            "current_topic": self.conversation_context.current_topic,
            "mentioned_entities": list(self.conversation_context.mentioned_entities),
            "last_intent": self.conversation_context.last_query_intent.intent_type if self.conversation_context.last_query_intent else None
        }
        return debug_info

    def interactive_loop(self) -> None:
        """Run interactive query loop."""
        print("✨ Enhanced RAG System Ready")
        print("💡 Commands: exit, debug:<query>, no-analysis:<query>, reset, summary, help")
        while True:
            try:
                query = input("🧠 Enhanced RAG > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
            if query.lower() in {"exit", "quit", "q"}:
                print("👋 Goodbye!")
                break
            if not query:
                continue
            if query.lower() == "help":
                print("💡 Commands:\n"
                      "   • Regular query: Type your question\n"
                      "   • debug:<query> - Show processing details\n"
                      "   • no-analysis:<query> - Query without analysis\n"
                      "   • reset - Clear conversation history\n"
                      "   • summary - Show conversation summary\n"
                      "   • exit/quit - Exit system")
                continue
            elif query.lower() == "reset":
                self.reset_conversation()
                continue
            elif query.lower() == "summary":
                print(self.get_conversation_summary())
                continue
            elif query.startswith("debug:"):
                actual_query = query[6:].strip()
                if actual_query:
                    debug_info = self.debug_query_processing(actual_query)
                    print("🔍 Debug Information:")
                    print(json.dumps(debug_info, indent=2, default=str))
                else:
                    print("❌ Please provide a query after 'debug:'")
                continue
            elif query.startswith("no-analysis:"):
                actual_query = query[12:].strip()
                if actual_query:
                    response = self.query(actual_query, include_analysis=False)
                    print(f"📝 {response}")
                else:
                    print("❌ Please provide a query after 'no-analysis:'")
                continue
            try:
                response = self.query(query)
                print(f"📝 {response}")
            except Exception as e:
                logger.error(f"Error processing query: {e}", query=query)
                print(f"❌ Error: {e}")
            print()

def main():
    """Main entry point with enhanced argument parsing."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Enhanced RAG Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python enhanced_rag.py --config config.yaml\n"
               "  python enhanced_rag.py --debug"
    )
    parser.add_argument('--config', help='Path to YAML config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.debug:
        structlog.configure(processors=[structlog.processors.JSONRenderer(indent=2)])

    try:
        rag = EnhancedRAG(args.config)
        rag.interactive_loop()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
