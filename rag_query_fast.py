#!/usr/bin/env python3

import os
import json
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout

from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "Documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

@dataclass
class DocumentPattern:
    """Represents a pattern for document classification."""
    name: str
    regex_patterns: List[str]
    keywords: List[str] = field(default_factory=list)
    path_hints: List[str] = field(default_factory=list)

@dataclass
class DocumentType:
    """Represents a type of document with its patterns and priority."""
    name: str
    description: str
    priority: int  # Higher number = higher priority
    patterns: List[DocumentPattern]
    confidence_base: float = 0.7

class DocumentClassifier:
    """Generic document classifier that can be configured for any domain."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.document_types = []
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._load_default_config()
    
    def load_config(self, config_path: str):
        """Load document classification config from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.document_types = []
        for doc_type_data in config.get('document_types', []):
            patterns = []
            for pattern_data in doc_type_data.get('patterns', []):
                patterns.append(DocumentPattern(
                    name=pattern_data['name'],
                    regex_patterns=pattern_data.get('regex_patterns', []),
                    keywords=pattern_data.get('keywords', []),
                    path_hints=pattern_data.get('path_hints', [])
                ))
            
            self.document_types.append(DocumentType(
                name=doc_type_data['name'],
                description=doc_type_data['description'],
                priority=doc_type_data.get('priority', 1),
                patterns=patterns,
                confidence_base=doc_type_data.get('confidence_base', 0.7)
            ))
        
        # Sort by priority (highest first)
        self.document_types.sort(key=lambda x: x.priority, reverse=True)
    
    def _load_default_config(self):
        """Load a generic default configuration."""
        # Personal documents (highest priority)
        personal_patterns = [
            DocumentPattern("personal_record", 
                          regex_patterns=[r"\bmy\s+", r"\bi\s+have\s+", r"\bmine\b"],
                          keywords=["personal", "diary", "journal", "log"]),
            DocumentPattern("receipt", 
                          regex_patterns=[r"receipt|invoice|bill|purchase"],
                          keywords=["paid", "bought", "purchased"]),
            DocumentPattern("certificate", 
                          regex_patterns=[r"certificate|diploma|license|permit"],
                          keywords=["certified", "issued", "valid"])
        ]
        
        # Reference materials (medium priority)
        reference_patterns = [
            DocumentPattern("manual", 
                          regex_patterns=[r"manual|guide|handbook|instructions"],
                          keywords=["how to", "step by step", "procedure"],
                          path_hints=["manual", "guide", "docs"]),
            DocumentPattern("specification", 
                          regex_patterns=[r"spec|specification|technical\s+data"],
                          keywords=["technical", "parameters", "features"]),
            DocumentPattern("catalog", 
                          regex_patterns=[r"catalog|catalogue|brochure"],
                          keywords=["products", "models", "available"])
        ]
        
        # General content (lowest priority)
        general_patterns = [
            DocumentPattern("article", 
                          regex_patterns=[r"article|blog|post|news"],
                          keywords=["published", "author", "date"]),
            DocumentPattern("note", 
                          regex_patterns=[r"note|memo|reminder"],
                          keywords=["remember", "todo", "note"])
        ]
        
        self.document_types = [
            DocumentType("personal_document", "Personal records and ownership evidence", 
                        3, personal_patterns, 0.9),
            DocumentType("reference_material", "Manuals, guides, and specifications", 
                        2, reference_patterns, 0.7),
            DocumentType("general_content", "Articles, notes, and other content", 
                        1, general_patterns, 0.5)
        ]
    
    def classify_document(self, content: str, path: str = "", title: str = "") -> Dict[str, Any]:
        """Classify a document based on content, path, and title."""
        import re
        
        content_lower = content.lower()
        path_lower = path.lower()
        title_lower = title.lower()
        
        best_classification = {
            "type": "unknown",
            "confidence": 0.0,
            "indicators": [],
            "description": "Unclassified document"
        }
        
        for doc_type in self.document_types:
            score = 0
            indicators = []
            
            for pattern in doc_type.patterns:
                pattern_score = 0
                
                # Check regex patterns
                for regex_pattern in pattern.regex_patterns:
                    if re.search(regex_pattern, content_lower, re.IGNORECASE):
                        pattern_score += 2
                        indicators.append(f"regex: {pattern.name}")
                        break
                
                # Check keywords
                for keyword in pattern.keywords:
                    if keyword.lower() in content_lower:
                        pattern_score += 1
                        indicators.append(f"keyword: {keyword}")
                
                # Check path hints
                for hint in pattern.path_hints:
                    if hint.lower() in path_lower or hint.lower() in title_lower:
                        pattern_score += 1
                        indicators.append(f"path: {hint}")
                
                score += pattern_score
            
            # Calculate confidence based on score and base confidence
            confidence = min(doc_type.confidence_base + (score * 0.1), 1.0)
            
            # Update best classification if this is better
            if confidence > best_classification["confidence"]:
                best_classification = {
                    "type": doc_type.name,
                    "confidence": confidence,
                    "indicators": indicators[:5],  # Limit indicators
                    "description": doc_type.description
                }
        
        return best_classification

def make_weaviate_client(url: str) -> weaviate.WeaviateClient:
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

# Generic prompt template 
GENERIC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant helping someone search their personal knowledge base.

Documents come in three loose priority bands:
- HIGH  : Personal records, receipts, ownership evidence
- MEDIUM: Manuals, reference specs, how-to guides
- LOW   : General notes, articles, miscellaneous content

Some chunks include extra metadata (e.g. folder path or tags).  
Feel free to quote it when it genuinely helps the user locate or trust the information, but it is never required.

Context:
{context}

Question: {question}

Instructions:
1. Prefer higher-priority documents when they answer the question.
2. Use reference materials to add detail or background.
3. Mention metadata such as folder or tags only when it adds clarity.
4. Distinguish clearly between what the user owns/has and general facts.
5. Cite specific evidence you found.

Answer:"""
)

class GenericRAG:
    def __init__(self, config_path: Optional[str] = None):
        self.client = make_weaviate_client(WEAVIATE_URL)
        assert self.client.is_ready(), "Weaviate instance is not ready"
        
        self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=WEAVIATE_INDEX,
            text_key="text",
            embedding=self.embedder,
        )
        
        self.llm = OllamaLLM(model=OLLAMA_MODEL)
        self.classifier = DocumentClassifier(config_path)
        
        # Create QA chain with generic prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 12}),
            chain_type_kwargs={"prompt": GENERIC_PROMPT}
        )
    
    def analyze_retrieved_docs(self, query: str) -> Dict[str, Any]:
        """Analyze retrieved documents with generic classification."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke(query)
        
        analysis = {
            "total_docs": len(docs),
            "by_type": {},
            "classifications": []
        }
        
        for doc in docs:
            content = doc.page_content
            path = doc.metadata.get("path", "")
            title = doc.metadata.get("title", "")
            
            classification = self.classifier.classify_document(content, path, title)
            
            doc_info = {
                "title": title,
                "path": path,
                "classification": classification,
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
            
            analysis["classifications"].append(doc_info)
            
            # Group by type
            doc_type = classification["type"]
            if doc_type not in analysis["by_type"]:
                analysis["by_type"][doc_type] = []
            
            analysis["by_type"][doc_type].append({
                "title": title,
                "path": path,
                "indicators": classification["indicators"],
                "confidence": classification["confidence"]
            })
        
        return analysis
    
    def query_with_analysis(self, query: str, show_analysis: bool = True) -> str:
        """Query with optional document analysis."""
        try:
            # Get the main answer
            result = self.qa_chain.invoke(query)
            answer = result["result"].strip()
            
            # Add analysis for ownership/personal questions
            if show_analysis and any(keyword in query.lower() for keyword in 
                                   ["do i have", "do i own", "what do i have", "my ", "mine", "personal"]):
                analysis = self.analyze_retrieved_docs(query)
                
                analysis_text = f"\n\n📊 Document Analysis:"
                analysis_text += f"\n• Total documents found: {analysis['total_docs']}"
                
                # Show breakdown by type
                for doc_type, docs in analysis['by_type'].items():
                    if docs:
                        analysis_text += f"\n• {doc_type.replace('_', ' ').title()}: {len(docs)}"
                
                # Show top documents by type
                for doc_type, docs in analysis['by_type'].items():
                    if docs and len(docs) > 0:
                        analysis_text += f"\n\n📁 {doc_type.replace('_', ' ').title()}:"
                        for doc in docs[:3]:  # Show top 3
                            indicators_str = ', '.join(doc['indicators'][:3]) if doc['indicators'] else 'general match'
                            analysis_text += f"\n  • {doc['title']}: {indicators_str}"
                
                answer += analysis_text
            
            return answer
            
        except Exception as e:
            return f"⚠️ Error during query: {e}"
    
    def save_default_config(self, config_path: str):
        """Save the current classification config to a JSON file."""
        config = {
            "document_types": []
        }
        
        for doc_type in self.classifier.document_types:
            patterns_data = []
            for pattern in doc_type.patterns:
                patterns_data.append({
                    "name": pattern.name,
                    "regex_patterns": pattern.regex_patterns,
                    "keywords": pattern.keywords,
                    "path_hints": pattern.path_hints
                })
            
            config["document_types"].append({
                "name": doc_type.name,
                "description": doc_type.description,
                "priority": doc_type.priority,
                "confidence_base": doc_type.confidence_base,
                "patterns": patterns_data
            })
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuration saved to {config_path}")
    
    def interactive_loop(self):
        """Interactive loop with generic commands."""
        print("✨ Connected to Generic RAG system")
        print("💡 Commands: 'exit', 'save-config <path>', 'debug:<query>', 'no-analysis:<query>'")
        
        while True:
            try:
                query = input("🧠 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                continue
            
            # Special commands
            if query.startswith("save-config"):
                path = query.split(None, 1)[1] if len(query.split()) > 1 else "rag_config.json"
                self.save_default_config(path)
                continue
            
            if query.startswith("debug:"):
                actual_query = query[6:].strip()
                analysis = self.analyze_retrieved_docs(actual_query)
                print(f"🔍 Debug Analysis for '{actual_query}':")
                for i, cls in enumerate(analysis["classifications"][:5]):
                    print(f"  {i+1}. {cls['title']}")
                    print(f"     Type: {cls['classification']['type']}")
                    print(f"     Confidence: {cls['classification']['confidence']:.2f}")
                    print(f"     Indicators: {cls['classification']['indicators']}")
                    print(f"     Preview: {cls['content_preview'][:100]}...")
                    print()
                continue
            
            # Query without analysis
            if query.startswith("no-analysis:"):
                actual_query = query[12:].strip()
                answer = self.query_with_analysis(actual_query, show_analysis=False)
                print("📝", answer)
                continue
            
            # Regular query with analysis
            answer = self.query_with_analysis(query)
            print("📝", answer)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generic RAG Query System')
    parser.add_argument('--config', help='Path to classification config JSON file')
    parser.add_argument('--save-default-config', help='Save default config to specified path and exit')
    
    args = parser.parse_args()
    
    if args.save_default_config:
        rag = GenericRAG()
        rag.save_default_config(args.save_default_config)
        return
    
    rag = GenericRAG(args.config)
    rag.interactive_loop()

if __name__ == "__main__":
    main()