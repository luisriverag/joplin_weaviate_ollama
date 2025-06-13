#!/usr/bin/env python3

import os
import json
import re
import hashlib
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter

from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---------------------------------------------------------------------------
# 🛠️  Environment ----------------------------------------------------------------
# ---------------------------------------------------------------------------

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "Documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# ---------------------------------------------------------------------------
# 🔧 Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 📐 Data classes --------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Represents the detected intent of a user query."""
    intent_type: str  # 'search', 'ownership', 'comparison', 'temporal', 'procedural'
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

    def add_exchange(self, query: str, response: str, intent: QueryIntent):
        """Add a query‑response exchange to history."""
        self.history.append({
            'query': query,
            'response': response,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        self.last_query_intent = intent

        # Extract and track entities
        entities = self._extract_entities(query)
        self.mentioned_entities.update(entities)

        # Update current topic
        if intent.intent_type in ['search', 'ownership']:
            self.current_topic = ' '.join(intent.keywords[:3])

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction – can be enhanced with NER."""
        entities = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', text)
        entities.extend(re.findall(r'"([^"]*)"', text))
        entities.extend(re.findall(r"'([^']*)'", text))
        return entities

    def get_context_summary(self) -> str:
        if not self.history:
            return ""
        recent = list(self.history)[-3:]
        parts = []
        for ex in recent:
            parts.append(f"Q: {ex['query'][:100]}")
            parts.append(f"A: {ex['response'][:100]}…")
        return "\n".join(parts)

# ---------------------------------------------------------------------------
# 📄 Document classifier (EN/ES) ---------------------------------------------
# ---------------------------------------------------------------------------

class EnhancedDocumentClassifier:
    """Advanced document classifier with bilingual support."""

    def __init__(self, config_path: Optional[str] = None):
        self.load_config(config_path)
        self._compile_patterns()

    # ��� Load configuration --------------------------------------------------
    def load_config(self, config_path: Optional[str]):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.patterns = config
            return

        # Default EN/ES patterns -------------------------------------------
        self.patterns = {
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
                {
                    "name": "technical_reference",
                    "description": "Technical manuals, specifications, and guides / Manuales técnicos, especificaciones y guías",
                    "priority": 8,
                    "patterns": {
                        "technical_terms": [
                            # English
                            r"\bspecification\b", r"\bmanual\b", r"\bguide\b", r"\bprotocol\b", r"\bstandard\b", r"\bAPI\b",
                            # Spanish
                            r"\bespecificaci(?:[óo])n\b", r"\bmanual\b", r"\bgu[íi]a\b", r"\bprotocolo\b", r"\best[áa]ndar\b", r"\binterfaz\s+de\s+programaci(?:[óo])n\b"
                        ],
                        "instruction_words": [
                            # English
                            r"\bhow\s+to\b", r"\bstep\s+by\s+step\b", r"\bprocedure\b", r"\binstructions\b",
                            # Spanish
                            r"\bc(?:ó|o)mo\s+\b", r"\bpaso\s+a\s+paso\b", r"\bprocedimiento\b", r"\binstrucciones\b"
                        ]
                    },
                    "boost_keywords": [
                        # English
                        "manual", "guide", "specification", "how to",
                        # Spanish
                        "manual", "guía", "especificación", "cómo"
                    ],
                    "temporal_importance": False
                },
                {
                    "name": "financial_record",
                    "description": "Financial documents and transactions / Documentos financieros y transacciones",
                    "priority": 9,
                    "patterns": {
                        "financial_terms": [
                            # English
                            r"\$\d+", r"\bprice\b", r"\bcost\b", r"\bpaid\b", r"\btransaction\b", r"\bexpense\b", r"\bbill\b",
                            # Spanish
                            r"€\d+", r"\bprecio\b", r"\bcoste?\b", r"\bpag(?:o|ado)\b", r"\btransacci(?:[óo])n\b", r"\bgasto\b", r"\bfactura\b"
                        ],
                        "account_info": [
                            # English
                            r"\baccount\s+\w+", r"\bbank\b", r"\bcredit\b",
                            # Spanish
                            r"\bcuenta\s+\w+", r"\bbanco\b", r"\bcr[ée]dito\b"
                        ]
                    },
                    "boost_keywords": [
                        # English
                        "paid", "transaction", "expense", "bill", "cost",
                        # Spanish
                        "pagado", "transacción", "gasto", "factura", "coste"
                    ],
                    "temporal_importance": True
                }
            ]
        }

    # 🔄 Compile regex patterns once ----------------------------------------
    def _compile_patterns(self):
        self.compiled_patterns = {}
        for dtype in self.patterns["document_types"]:
            name = dtype["name"]
            self.compiled_patterns[name] = {}
            for group, patterns in dtype["patterns"].items():
                self.compiled_patterns[name][group] = [re.compile(p, re.IGNORECASE) for p in patterns]

    # 🏷️  Classify -------------------------------------------------------------
    def classify_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        path = metadata.get("path", "")
        title = metadata.get("title", "")
        tags = metadata.get("tags", [])
        last_modified = metadata.get("last_modified", "")

        content_lower = content.lower()
        path_lower = path.lower()
        title_lower = title.lower()

        classifications = []

        for dtype in self.patterns["document_types"]:
            name = dtype["name"]
            score = 0
            matched = []
            for group, regexes in self.compiled_patterns[name].items():
                gm = 0
                for rgx in regexes:
                    if rgx.search(content):
                        gm += 1
                        matched.append(f"{group}:{rgx.pattern[:30]}")
                # group weighting
                weight = 3 if group == "ownership_indicators" else 2 if group == "document_types" else 1
                score += gm * weight
            # Keyword boosts
            for kw in dtype.get("boost_keywords", []):
                if kw in content_lower:
                    score += 1
                if kw in title_lower:
                    score += 2
                if kw in tags:
                    score += 1.5
            # Path hints
            if any(h in path_lower for h in ["recibo", "receipt", "manual", "guía", "guide", "doc"]):
                score += 1
            # Temporal boost
            if dtype.get("temporal_importance") and last_modified:
                try:
                    mod_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    days_old = (datetime.now() - mod_date.replace(tzinfo=None)).days
                    score += 2 if days_old < 30 else 1 if days_old < 90 else 0
                except Exception:
                    pass
            confidence = min(score * 0.15, 1.0)
            classifications.append({
                "type": name,
                "description": dtype["description"],
                "confidence": confidence,
                "score": score,
                "priority": dtype["priority"],
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

# ---------------------------------------------------------------------------
# 🔍 Intent detector (EN/ES) --------------------------------------------------
# ---------------------------------------------------------------------------

class QueryIntentDetector:
    """Detects the intent behind user queries in English or Spanish."""

    def __init__(self):
        self.intent_patterns = {
            "ownership": [
                # English
                r"\bdo\s+i\s+(have|own|possess)\b", r"\bwhat\s+do\s+i\s+(have|own|possess)\b", r"\bmy\s+\w+", r"\bmine\b", r"\bshow\s+me\s+what\s+i\b",
                # Spanish
                r"\b(tengo|poseo|dispongo)\b", r"\bqu[ée]\s+(tengo|poseo)\b", r"\bmi\s+\w+", r"\bm(?:í|i)o\b"
            ],
            "search": [
                # English
                r"\bfind\b", r"\bsearch\b", r"\blook\s+for\b", r"\bshow\s+me\b", r"\bwhere\s+is\b",
                # Spanish
                r"\bbuscar\b", r"\bencuentra?\b", r"\bd[óo]nde\s+est[áa]\b", r"\bmuestra?me\b", r"\benséñame\b"
            ],
            "comparison": [
                # English
                r"\bcompare\b", r"\bdifference\b", r"\bvs\b", r"\bbetter\b", r"\bworse\b", r"\bwhich\s+is\b",
                # Spanish
                r"\bcompara?r?\b", r"\bdiferencia\b", r"\bmejor\b", r"\bpeor\b", r"\bcu[aá]l\s+es\b"
            ],
            "temporal": [
                # English
                r"\bwhen\b", r"\brecent\b", r"\blast\s+\w+\b", r"\btoday\b", r"\byesterday\b", r"\bthis\s+\w+\b", r"\bsince\b", r"\buntil\b", r"\bbefore\b", r"\bafter\b",
                # Spanish
                r"\bcu[aá]ndo\b", r"\breciente\b", r"\b(la|el)\s+\w+\s+pasad[ao]\b", r"\bhoy\b", r"\bayer\b", r"\besta\s+\w+\b", r"\bdesde\b", r"\bhasta\b", r"\bantes\b", r"\bdespu[eé]s\b"
            ],
            "procedural": [
                # English
                r"\bhow\s+to\b", r"\bsteps?\b", r"\bprocess\b", r"\bprocedure\b", r"\bmethod\b", r"\bway\s+to\b",
                # Spanish
                r"\bc[óo]mo\b", r"\bpasos?\b", r"\bproceso\b", r"\bprocedimiento\b", r"\bm[ée]todo\b", r"\bforma\s+de\b"
            ]
        }
        # Pre‑compile
        self.compiled_patterns = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in self.intent_patterns.items()}

    # 🕵️‍♀️ Detect ----------------------------------------------------------
    def detect_intent(self, query: str) -> QueryIntent:
        q_lower = query.lower()
        scores = defaultdict(float)
        matched_kw = defaultdict(list)
        for intent, pats in self.compiled_patterns.items():
            for pat in pats:
                hits = pat.findall(query)
                if hits:
                    scores[intent] += len(hits)
                    matched_kw[intent].extend(hits if isinstance(hits[0], str) else [h[0] for h in hits])
        # Temporal context extraction (EN/ES)
        temporal_patterns = [
            # absolute temporal markers
            r"\b(today|hoy|yesterday|ayer|last\s+week|la\s+semana\s+pasada|last\s+month|el\s+mes\s+pasado|this\s+year|este\s+a[ñn]o)\b",
            # relative before/after
            r"\b(before|after|antes|despu[eé]s|since|desde|until|hasta)\s+(\w+)\b",
            # dates
            r"\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})\b"
        ]
        temporal_context = None
        for p in temporal_patterns:
            m = re.search(p, q_lower)
            if m:
                temporal_context = m.group(0)
                break
        if scores:
            primary = max(scores, key=scores.get)
            conf = min(scores[primary] * 0.3, 1.0)
        else:
            primary = "search"
            conf = 0.3
        # Extract keywords (both languages, >2 letters)
        kws = re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]{3,}\b", query)
        stop = {
            'the', 'and', 'for', 'are', 'have', 'with', 'this', 'that',
            'los', 'las', 'unos', 'unas', 'para', 'con', 'este', 'esta', 'ese', 'esa', 'son', 'tengo'
        }
        kws = [k for k in kws if k.lower() not in stop][:10]
        return QueryIntent(primary, conf, kws, temporal_context, self._extract_entities(query))

    # Entity extraction helper ---------------------------------------------
    def _extract_entities(self, txt: str) -> List[str]:
        entities = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*\b', txt)
        entities.extend(re.findall(r'"([^"]*)"', txt))
        entities.extend(re.findall(r"'([^']*)'", txt))
        entities.extend(re.findall(r'\b\d+[A-Za-z]+\b', txt))
        return entities

# ---------------------------------------------------------------------------
# 📚 Enhanced Retriever -------------------------------------------------------
# ---------------------------------------------------------------------------

class EnhancedRetriever:
    """Retrieval layer aware of bilingual patterns."""

    def __init__(self, vectorstore: WeaviateVectorStore, client: weaviate.WeaviateClient, index_name: str, embedder: HuggingFaceEmbeddings):
        self.vectorstore = vectorstore
        self.client = client
        self.collection = client.collections.get(index_name)
        self.embedder = embedder

    # 🗂️  Main retrieve ------------------------------------------------------
    def retrieve_with_strategy(self, query: str, intent: QueryIntent, ctx: ConversationContext, k: int = 12) -> List[Document]:
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

    # 🔍 Vector search -------------------------------------------------------
    def _vector_search(self, query: str, k: int):
        return self.vectorstore.similarity_search(query, k=k)

    # 🏠 Ownership filter (EN/ES) -------------------------------------------
    def _filter_ownership_docs(self, docs: List[Document]):
        indicators = [
            # English
            "purchased", "bought", "own", "mine", "receipt", "invoice", "warranty", "certificate", "my", "belongs to",
            # Spanish
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
        selected.sort(key=lambda x: x.metadata.get("ownership_score", 0), reverse=True)
        return selected or docs

    # ⏳ Temporal filter -----------------------------------------------------
    def _filter_temporal_docs(self, docs: List[Document], intent: QueryIntent):
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
            except Exception:
                continue
        return out if out else docs

    # 📚 Procedural boost (EN/ES) -------------------------------------------
    def _boost_procedural_docs(self, docs: List[Document]):
        indicators = [
            # English
            "step", "procedure", "method", "process", "how to", "instructions", "guide", "manual", "tutorial",
            # Spanish
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
        docs.sort(key=lambda x: x.metadata.get("procedural_score", 0), reverse=True)
        return docs

    # 🤝 Conversation context boost ----------------------------------------
    def _apply_conversation_context(self, docs: List[Document], ctx: ConversationContext):
        if not ctx.mentioned_entities:
            return docs
        for d in docs:
            cscore = 0
            for ent in ctx.mentioned_entities:
                if ent.lower() in d.page_content.lower():
                    cscore += 1
            if ctx.current_topic:
                for w in ctx.current_topic.split():
                    if w.lower() in d.page_content.lower():
                        cscore += 0.5
            d.metadata["context_score"] = cscore
        return docs

    # 🔀 Re‑rank -------------------------------------------------------------
    def _rerank_documents(self, docs: List[Document], query: str, intent: QueryIntent):
        q_words = set(query.lower().split())
        for d in docs:
            score = 1.0
            if intent.intent_type == "ownership":
                score += d.metadata.get("ownership_score", 0) * 0.5
            elif intent.intent_type == "procedural":
                score += d.metadata.get("procedural_score", 0) * 0.3
            score += d.metadata.get("temporal_relevance", 0) * 0.4
            score += d.metadata.get("context_score", 0) * 0.2
            # keyword overlap
            d_words = set(d.page_content.lower().split())
            overlap = len(q_words & d_words) / (len(q_words) or 1)
            score += overlap * 0.3
            d.metadata["final_relevance_score"] = score
        docs.sort(key=lambda x: x.metadata.get("final_relevance_score", 0), reverse=True)
        return docs


class EnhancedRAG:
    """Enhanced RAG system with advanced capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.client = self._make_weaviate_client(WEAVIATE_URL)
        assert self.client.is_ready(), "Weaviate instance is not ready"
        
        self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=WEAVIATE_INDEX,
            text_key="text",
            embedding=self.embedder,
        )
        
        self.llm = OllamaLLM(model=OLLAMA_MODEL)
        self.classifier = EnhancedDocumentClassifier(config_path)
        self.intent_detector = QueryIntentDetector()
        self.retriever = EnhancedRetriever(self.vectorstore, self.client, WEAVIATE_INDEX, self.embedder)
        self.conversation_context = ConversationContext()
        
        # Enhanced prompt template
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
        
        # Create enhanced QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 1}),  # Dummy, we'll override
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    def _make_weaviate_client(self, url: str) -> weaviate.WeaviateClient:
        """Create Weaviate v4 client from URL."""
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        
        if host in {"localhost", "127.0.0.1"}:
            return weaviate.connect_to_local(
                host=host, port=port, grpc_port=50051,
                additional_config=AdditionalConfig(timeout=Timeout(init=5))
            )
        else:
            return weaviate.connect_to_custom(
                http_host=host, http_port=port, http_secure=(parsed.scheme == "https"),
                grpc_host=host, grpc_port=50051, grpc_secure=(parsed.scheme == "https"),
                additional_config=AdditionalConfig(timeout=Timeout(init=5))
            )
    
    def query(self, query: str, include_analysis: bool = None) -> str:
        """Enhanced query processing with intent detection and context awareness."""
        try:
            # Detect query intent
            intent = self.intent_detector.detect_intent(query)
            logger.info(f"Detected intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
            
            # Auto-decide whether to include analysis
            if include_analysis is None:
                include_analysis = intent.intent_type in ["ownership", "comparison"] or intent.confidence > 0.7
            
            # Enhanced retrieval
            relevant_docs = self.retriever.retrieve_with_strategy(
                query, intent, self.conversation_context, k=12
            )
            
            # Prepare context
            context = self._prepare_context(relevant_docs)
            conversation_history = self.conversation_context.get_context_summary()
            
            # Format intent for prompt
            intent_str = f"Type: {intent.intent_type}, Confidence: {intent.confidence:.2f}, Keywords: {', '.join(intent.keywords[:5])}"
            if intent.temporal_context:
                intent_str += f", Temporal: {intent.temporal_context}"
            
            # Generate response
            response = self.llm.invoke(
                self.prompt_template.format(
                    context=context,
                    question=query,
                    intent=intent_str,
                    conversation_history=conversation_history
                )
            )
            
            # Add analysis if requested
            if include_analysis:
                analysis = self._generate_analysis(relevant_docs, intent, query)
                response += f"\n\n{analysis}"
            
            # Update conversation context
            self.conversation_context.add_exchange(query, response, intent)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"⚠️ Error processing query: {e}"
    
    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(docs):
            # Add document info
            title = doc.metadata.get("title", f"Document {i+1}")
            path = doc.metadata.get("path", "")
            tags = doc.metadata.get("tags", [])
            
            context_part = f"[Document {i+1}: {title}"
            if path:
                context_part += f" | Path: {path}"
            if tags:
                context_part += f" | Tags: {', '.join(tags)}"
            
            # Add relevance score if available
            if "final_relevance_score" in doc.metadata:
                context_part += f" | Relevance: {doc.metadata['final_relevance_score']:.2f}"
            
            context_part += f"]\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _generate_analysis(self, docs: List[Document], intent: QueryIntent, query: str) -> str:
        """Generate analysis of retrieved documents."""
        analysis_parts = ["📊 Document Analysis:"]
        
        # Basic stats
        analysis_parts.append(f"• Found {len(docs)} relevant documents")
        analysis_parts.append(f"• Query intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        
        # Document type breakdown
        doc_types = defaultdict(int)
        high_confidence_docs = 0
        
        for doc in docs:
            # Count document types
            if "classification" in doc.metadata:
                classification = doc.metadata["classification"]
                doc_type = classification.get("type", "unknown")
                doc_types[doc_type] += 1
                
                if classification.get("confidence", 0) > 0.7:
                    high_confidence_docs += 1
            else:
                doc_types["unclassified"] += 1
        
        # Show document type breakdown
        if doc_types:
            analysis_parts.append("• Document types found:")
            for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
                display_name = doc_type.replace("_", " ").title()
                analysis_parts.append(f"  - {display_name}: {count}")
        
        # High-confidence matches
        if high_confidence_docs > 0:
            analysis_parts.append(f"• High-confidence matches: {high_confidence_docs}")
        
        # Intent-specific analysis
        if intent.intent_type == "ownership":
            ownership_docs = [doc for doc in docs if doc.metadata.get("ownership_score", 0) > 0]
            if ownership_docs:
                analysis_parts.append(f"• Documents with ownership indicators: {len(ownership_docs)}")
                # Show top ownership documents
                ownership_docs.sort(key=lambda d: d.metadata.get("ownership_score", 0), reverse=True)
                for doc in ownership_docs[:3]:
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
        
        # Show top-ranked documents
        if len(docs) > 0:
            analysis_parts.append("• Top relevant documents:")
            for i, doc in enumerate(docs[:3]):
                title = doc.metadata.get("title", f"Document {i+1}")
                relevance = doc.metadata.get("final_relevance_score", 0)
                analysis_parts.append(f"  {i+1}. {title} (relevance: {relevance:.2f})")
        
        return "\n".join(analysis_parts)
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.conversation_context.history:
            return "No conversation history."
        
        summary_parts = ["🔄 Conversation Summary:"]
        summary_parts.append(f"• Total exchanges: {len(self.conversation_context.history)}")
        
        if self.conversation_context.current_topic:
            summary_parts.append(f"• Current topic: {self.conversation_context.current_topic}")
        
        if self.conversation_context.mentioned_entities:
            entities = list(self.conversation_context.mentioned_entities)
            summary_parts.append(f"• Mentioned entities: {', '.join(entities[:5])}")
        
        # Show recent queries
        summary_parts.append("• Recent queries:")
        recent = list(self.conversation_context.history)[-3:]
        for exchange in recent:
            intent = exchange.get("intent", {})
            intent_type = intent.intent_type if hasattr(intent, 'intent_type') else "unknown"
            summary_parts.append(f"  - {exchange['query'][:50]}... (intent: {intent_type})")
        
        return "\n".join(summary_parts)
    
    def reset_conversation(self):
        """Reset the conversation context."""
        self.conversation_context = ConversationContext()
        print("🔄 Conversation context reset.")
    
    def debug_query_processing(self, query: str) -> Dict[str, Any]:
        """Debug query processing pipeline."""
        debug_info = {}
        
        # Intent detection
        intent = self.intent_detector.detect_intent(query)
        debug_info["intent"] = {
            "type": intent.intent_type,
            "confidence": intent.confidence,
            "keywords": intent.keywords,
            "temporal_context": intent.temporal_context,
            "entities": intent.entities
        }
        
        # Document retrieval
        docs = self.retriever.retrieve_with_strategy(
            query, intent, self.conversation_context, k=5
        )
        
        debug_info["retrieval"] = {
            "num_docs": len(docs),
            "documents": []
        }
        
        for doc in docs:
            doc_info = {
                "title": doc.metadata.get("title", "Untitled"),
                "path": doc.metadata.get("path", ""),
                "relevance_score": doc.metadata.get("final_relevance_score", 0),
                "ownership_score": doc.metadata.get("ownership_score", 0),
                "procedural_score": doc.metadata.get("procedural_score", 0),
                "context_score": doc.metadata.get("context_score", 0),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            
            if "classification" in doc.metadata:
                classification = doc.metadata["classification"]
                doc_info["classification"] = {
                    "type": classification.get("type"),
                    "confidence": classification.get("confidence"),
                    "matched_patterns": classification.get("matched_patterns", [])
                }
            
            debug_info["retrieval"]["documents"].append(doc_info)
        
        # Conversation context
        debug_info["conversation_context"] = {
            "history_length": len(self.conversation_context.history),
            "current_topic": self.conversation_context.current_topic,
            "mentioned_entities": list(self.conversation_context.mentioned_entities),
            "last_intent": self.conversation_context.last_query_intent.intent_type if self.conversation_context.last_query_intent else None
        }
        
        return debug_info
    
    def interactive_loop(self):
        """Enhanced interactive loop with advanced commands."""
        print("✨ Enhanced RAG System Ready")
        print("💡 Commands:")
        print("   • 'exit' / 'quit' - Exit the system")
        print("   • 'debug:<query>' - Debug query processing")
        print("   • 'no-analysis:<query>' - Query without analysis")
        print("   • 'reset' - Reset conversation context")
        print("   • 'summary' - Show conversation summary")
        print("   • 'help' - Show this help message")
        print()
        
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
            
            # Handle special commands
            if query.lower() == "help":
                print("💡 Available commands:")
                print("   • Regular queries: Just type your question")
                print("   • debug:<query> - Show detailed processing information")
                print("   • no-analysis:<query> - Get answer without document analysis")
                print("   • reset - Clear conversation history")
                print("   • summary - Show conversation summary")
                print("   • exit/quit - Exit the system")
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
            
            # Regular query processing
            try:
                response = self.query(query)
                print(f"📝 {response}")
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ Error: {e}")
            
            print()  # Add spacing between interactions

def main():
    """Main entry point with enhanced argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced RAG Query System with Advanced Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_rag.py                          # Start interactive mode
  python enhanced_rag.py --config custom.json    # Use custom configuration
  python enhanced_rag.py --debug                  # Enable debug logging
        """
    )
    
    parser.add_argument('--config', 
                       help='Path to enhanced classification config JSON file')
    parser.add_argument('--debug', 
                       action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=getattr(logging, args.log_level))
    
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
