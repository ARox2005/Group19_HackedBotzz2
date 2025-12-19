"""
Text chunking module for splitting documents into smaller, overlapping chunks.
Supports configurable chunk size and overlap for optimal retrieval.
"""

from typing import List, Dict, Any, Optional
import re


class TextChunker:
    """
    Text chunker that splits documents into overlapping chunks.
    Uses sentence-aware splitting to avoid breaking mid-sentence.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 512)
            chunk_overlap: Number of characters to overlap between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Sentence-ending patterns
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split by sentence endings
        sentences = self.sentence_endings.split(text)
        
        # Clean up empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        chunks = []
        sentences = self.split_into_sentences(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence is longer than chunk size, split by characters
            if sentence_length > self.chunk_size:
                # First, save current chunk if exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence into character-based chunks
                for i in range(0, sentence_length, self.chunk_size - self.chunk_overlap):
                    chunk_end = min(i + self.chunk_size, sentence_length)
                    chunks.append(sentence[i:chunk_end])
                continue
            
            # Check if adding sentence would exceed chunk size
            potential_length = current_length + sentence_length + (1 if current_chunk else 0)
            
            if potential_length > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                # Take last few sentences that fit in overlap
                overlap_text = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) + 1 <= self.chunk_overlap:
                        overlap_text.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break
                
                current_chunk = overlap_text + [sentence]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk.append(sentence)
                current_length = potential_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document and preserve metadata for each chunk.
        
        Args:
            document: Document dictionary with 'text' and metadata
            
        Returns:
            List of chunk dictionaries with preserved metadata
        """
        text = document.get('text', '')
        chunks = self.chunk_text(text)
        
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'text': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source': document.get('source', ''),
                'filename': document.get('filename', ''),
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunk dictionaries
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks


class RecursiveTextChunker:
    """
    Alternative chunker that recursively splits text using different separators.
    Tries to keep semantic units together.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separators in order of preference (larger to smaller units)
        self.separators = [
            '\n\n\n',  # Multiple newlines (sections)
            '\n\n',    # Paragraph breaks
            '\n',      # Line breaks
            '. ',      # Sentences
            ', ',      # Clauses
            ' ',       # Words
            ''         # Characters
        ]
    
    def split_text(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of chunks
        """
        if separators is None:
            separators = self.separators.copy()
        
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # If text is small enough, return it
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split
            splits = list(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + len(separator)
            
            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    merged = separator.join(current_chunk)
                    
                    # If merged chunk is still too long, recurse
                    if len(merged) > self.chunk_size:
                        chunks.extend(self.split_text(merged, remaining_separators))
                    else:
                        chunks.append(merged)
                
                current_chunk = [split] if split.strip() else []
                current_length = len(split) if split.strip() else 0
            else:
                if split.strip():
                    current_chunk.append(split)
                    current_length += split_length
        
        # Handle remaining
        if current_chunk:
            merged = separator.join(current_chunk)
            if len(merged) > self.chunk_size:
                chunks.extend(self.split_text(merged, remaining_separators))
            else:
                chunks.append(merged)
        
        return [c for c in chunks if c.strip()]
    
    def chunk_text(self, text: str) -> List[str]:
        """Main method to chunk text."""
        return self.split_text(text)
