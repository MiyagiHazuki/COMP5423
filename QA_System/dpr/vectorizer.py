import os
import numpy as np
import requests
import time
import asyncio
import aiohttp
from typing import List, Union, Dict, Tuple
import json
from tqdm import tqdm

class doc_vectorizer:
    '''
    input:
    doc_id: document ids List[str]
    context: document texts List[str]
    api_key: silicone api key
    model_name: silicone model name

    output:
    doc_id: document ids List[str]
    vector: document vectors List[List[float32]]
    '''
    def __init__(self, doc_ids: List[str], contexts: List[str], api_key: str, model_name: str):
        self.doc_ids = doc_ids
        self.contexts = contexts
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.batch_size = 32
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def _make_request(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
        """
        Send synchronous requests to SILICONFLOW API and get embedding vectors
        
        Args:
            texts: A single text string or a list of strings
            batch_size: Batch size
            
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        # Process requests in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            payload = {
                "model": self.model_name,
                "input": batch_texts,
                "encoding_format": "float"
            }
            
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    print(f"Sending API request, batch size: {len(batch_texts)}, attempt {retry_count+1}...")
                    # Add request timeout
                    response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract embedding vectors
                    if len(batch_texts) == 1:
                        embeddings = [data['data'][0]['embedding']]
                    else:
                        # Sort by index to ensure consistency
                        sorted_data = sorted(data['data'], key=lambda x: x['index'])
                        embeddings = [item['embedding'] for item in sorted_data]
                    
                    all_embeddings.extend(embeddings)
                    print(f"API request successful, received {len(embeddings)} vectors, each with dimension {len(embeddings[0])}")
                    break
                    
                except requests.RequestException as e:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        print(f"Batch {i} to {i+batch_size} API request failed after {retry_count} retries: {e}")
                        if hasattr(e, 'response') and e.response:
                            print(f"Response content: {e.response.text}")
                        raise
                    else:
                        wait_time = self.retry_delay * (2 ** (retry_count - 1))  # exponential backoff
                        print(f"Batch {i} to {i+batch_size} request failed, waiting {wait_time} seconds before retrying ({retry_count}/{self.max_retries})...")
                        print(f"Error message: {str(e)}")
                        time.sleep(wait_time)
                
        return all_embeddings

    async def _async_make_request(self, session, batch_texts, batch_idx):
        """Asynchronously send requests and get embedding vectors"""
        payload = {
            "model": self.model_name,
            "input": batch_texts,
            "encoding_format": "float"
        }
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"API returned error code: {response.status}, response content: {text}")
                    
                    data = await response.json()
                    
                    # Extract embedding vectors
                    if len(batch_texts) == 1:
                        embeddings = [data['data'][0]['embedding']]
                    else:
                        # Sort by index to ensure consistency
                        sorted_data = sorted(data['data'], key=lambda x: x['index'])
                        embeddings = [item['embedding'] for item in sorted_data]
                    
                    return batch_idx, embeddings
            
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"Batch {batch_idx} API request failed after {retry_count} retries: {e}")
                    raise
                else:
                    print(f"Batch {batch_idx} request failed, retrying ({retry_count}/{self.max_retries})...")
                    await asyncio.sleep(self.retry_delay)

    async def _process_batches_async(self, max_concurrent_requests=5):
        """Process multiple batches of documents concurrently"""
        if not self.contexts:
            return []
        
        results = [None] * ((len(self.contexts) + self.batch_size - 1) // self.batch_size)
        
        async with aiohttp.ClientSession() as session:
            # Create task queue
            tasks = []
            
            batch_count = (len(self.contexts) + self.batch_size - 1) // self.batch_size
            with tqdm(total=batch_count, desc="Vectorization progress") as pbar:
                for i in range(0, len(self.contexts), self.batch_size):
                    batch_texts = self.contexts[i:i+self.batch_size]
                    batch_idx = i // self.batch_size
                    
                    # Control concurrency
                    while len(tasks) >= max_concurrent_requests:
                        # Wait for a task to complete
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        # Update task list
                        tasks = list(pending)
                        for task in done:
                            try:
                                idx, embeddings = await task
                                results[idx] = embeddings
                                pbar.update(1)
                            except Exception as e:
                                print(f"Error processing batch: {e}")
                    
                    # Add new task
                    task = asyncio.create_task(self._async_make_request(session, batch_texts, batch_idx))
                    tasks.append(task)
                
                # Wait for all remaining tasks to complete
                while tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    # Update task list
                    tasks = list(pending)
                    for task in done:
                        try:
                            idx, embeddings = await task
                            results[idx] = embeddings
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error processing batch: {e}")
        
        # Flatten results
        all_embeddings = []
        for batch in results:
            if batch:
                all_embeddings.extend(batch)
        
        return all_embeddings

    def vectorize(self, use_async=True, max_concurrent_requests=5) -> Tuple[List[str], List[List[float]]]:
        """
        Vectorize documents
        
        Args:
            use_async: Whether to use asynchronous processing
            max_concurrent_requests: Maximum concurrent requests
            
        Returns:
            tuple: (doc_ids, vectors)
        """
        print(f"Starting to process {len(self.contexts)} documents...")
        
        # Get embedding vectors
        if use_async:
            # Use asynchronous processing
            vectors = asyncio.run(self._process_batches_async(max_concurrent_requests))
        else:
            # Use synchronous processing
            vectors = self._make_request(self.contexts, self.batch_size)
        
        # Ensure doc_ids and vectors have the same length
        if len(self.doc_ids) != len(vectors):
            raise ValueError(f"Number of document IDs ({len(self.doc_ids)}) doesn't match number of vectors ({len(vectors)})")
            
        # L2 normalize vectors
        print("Performing vector normalization...")
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        zero_mask = norms == 0
        norms[zero_mask] = 1.0  # Avoid division by zero
        vectors = vectors / norms
        
        print(f"Vectorization complete! Got {len(vectors)} vectors.")
        return self.doc_ids, vectors.tolist()

    def process_large_dataset(self, chunk_size=1000, save_path=None):
        """
        Process large datasets in chunks, with option to save intermediate results
        
        Args:
            chunk_size: Size of each chunk
            save_path: Path format for saving intermediate results, e.g., 'vectors_{}.json'
            
        Returns:
            tuple: (doc_ids, vectors)
        """
        all_doc_ids = []
        all_vectors = []
        
        # Calculate number of chunks
        total_chunks = (len(self.doc_ids) + chunk_size - 1) // chunk_size
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(self.doc_ids))
            
            print(f"Processing chunk {i+1}/{total_chunks} (documents {start_idx} to {end_idx})...")
            
            # Get data for current chunk
            chunk_doc_ids = self.doc_ids[start_idx:end_idx]
            chunk_contexts = self.contexts[start_idx:end_idx]
            
            # Create vectorizer for current chunk
            chunk_vectorizer = doc_vectorizer(
                chunk_doc_ids, 
                chunk_contexts, 
                self.api_key, 
                self.model_name
            )
            
            # Vectorize current chunk
            chunk_doc_ids, chunk_vectors = chunk_vectorizer.vectorize()
            
            # Add to results
            all_doc_ids.extend(chunk_doc_ids)
            all_vectors.extend(chunk_vectors)
            
            # Save intermediate results
            if save_path:
                chunk_result = {
                    "doc_ids": chunk_doc_ids,
                    "vectors": chunk_vectors
                }
                with open(save_path.format(i), 'w', encoding='utf-8') as f:
                    json.dump(chunk_result, f)
                print(f"Results for chunk {i+1} saved to {save_path.format(i)}")
        
        return all_doc_ids, all_vectors

    def estimate_tokens(self, text):
        """
        Estimate the number of tokens in text (rough estimate: about 1.3 tokens per word)
        """
        # Rough estimate: English averages about 1.3 tokens per word, Chinese about 1-2 tokens per character
        words = len(text.split())
        chars = len(text) - words  # Non-space characters, rough calculation for Chinese characters
        return int(words * 1.3 + chars * 1.5)  # Rough estimate
    
    def split_long_text_with_overlap(self, text, max_tokens=7600, overlap_ratio=0.2):
        """
        Split long text into shorter pieces with overlap, each piece with tokens not exceeding max_tokens
        
        Args:
            text: The text to split
            max_tokens: Maximum tokens per segment
            overlap_ratio: Overlap ratio, default 20%
            
        Returns:
            List of split text segments
        """
        # If estimated tokens is less than limit, return directly
        est_tokens = self.estimate_tokens(text)
        if est_tokens <= max_tokens:
            return [text]
            
        print(f"Text length estimated at {est_tokens} tokens, needs splitting (with overlap)")
        
        # Calculate overlap tokens
        overlap_tokens = int(max_tokens * overlap_ratio)
        effective_tokens = max_tokens - overlap_tokens
        
        # Split by sentences
        sentences = text.replace('\n', '. ').split('. ')
        chunks = []
        current_sentences = []
        current_tokens = 0
        last_overlap_sentences = []
        last_overlap_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If a single sentence exceeds effective token limit, need further splitting
            if sentence_tokens > effective_tokens:
                # If current chunk is not empty, save it first
                if current_sentences:
                    chunks.append('. '.join(current_sentences))
                    # Record overlap part
                    overlap_count = 0
                    last_overlap_sentences = []
                    last_overlap_tokens = 0
                    for s in reversed(current_sentences):
                        s_tokens = self.estimate_tokens(s)
                        if last_overlap_tokens + s_tokens <= overlap_tokens:
                            last_overlap_sentences.insert(0, s)
                            last_overlap_tokens += s_tokens
                        else:
                            break
                    current_sentences = last_overlap_sentences.copy()
                    current_tokens = last_overlap_tokens
                
                # Split long sentences by words
                words = sentence.split()
                sub_chunk = []
                sub_tokens = 0
                
                for word in words:
                    word_tokens = self.estimate_tokens(word)
                    if sub_tokens + word_tokens > effective_tokens:
                        if sub_chunk:
                            # Add overlap part from previous chunk
                            if last_overlap_sentences and chunks:
                                full_chunk = '. '.join(last_overlap_sentences) + ' ' + ' '.join(sub_chunk)
                            else:
                                full_chunk = ' '.join(sub_chunk)
                            chunks.append(full_chunk)
                            
                            # Save overlap part
                            overlap_words = []
                            overlap_word_tokens = 0
                            for w in reversed(sub_chunk):
                                w_tokens = self.estimate_tokens(w)
                                if overlap_word_tokens + w_tokens <= overlap_tokens:
                                    overlap_words.insert(0, w)
                                    overlap_word_tokens += w_tokens
                                else:
                                    break
                            
                            sub_chunk = overlap_words.copy()
                            sub_tokens = overlap_word_tokens
                            last_overlap_sentences = []
                            last_overlap_tokens = 0
                    
                    sub_chunk.append(word)
                    sub_tokens += word_tokens
                
                if sub_chunk:
                    if last_overlap_sentences and chunks:
                        full_chunk = '. '.join(last_overlap_sentences) + ' ' + ' '.join(sub_chunk)
                    else:
                        full_chunk = ' '.join(sub_chunk)
                    chunks.append(full_chunk)
                    
                    # Update current chunk with overlap part
                    current_sentences = []
                    current_tokens = 0
                    last_overlap_sentences = []
                    last_overlap_tokens = 0
                    for w in reversed(sub_chunk):
                        w_tokens = self.estimate_tokens(w)
                        if last_overlap_tokens + w_tokens <= overlap_tokens:
                            last_overlap_sentences.insert(0, w)
                            last_overlap_tokens += w_tokens
                        else:
                            break
                    current_sentences = last_overlap_sentences.copy()
                    current_tokens = last_overlap_tokens
            
            # If adding this sentence would exceed effective limit, save current chunk and start new chunk
            elif current_tokens + sentence_tokens > effective_tokens:
                if current_sentences:
                    chunks.append('. '.join(current_sentences))
                    
                    # Save overlap part
                    last_overlap_sentences = []
                    last_overlap_tokens = 0
                    for s in reversed(current_sentences):
                        s_tokens = self.estimate_tokens(s)
                        if last_overlap_tokens + s_tokens <= overlap_tokens:
                            last_overlap_sentences.insert(0, s)
                            last_overlap_tokens += s_tokens
                        else:
                            break
                            
                    current_sentences = last_overlap_sentences.copy()
                    current_tokens = last_overlap_tokens
                
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
            
            # Otherwise add to current chunk
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Save the last chunk
        if current_sentences:
            chunks.append('. '.join(current_sentences))
            
        print(f"Text split into {len(chunks)} segments with overlap")
        return chunks
    
    def process_document_with_overlap(self, doc_id, text, overlap_ratio=0.2, use_weighted_avg=True):
        """
        Process a single document, automatically split long documents with overlap
        
        Args:
            doc_id: Document ID
            text: Document text
            overlap_ratio: Overlap ratio, default 20%
            use_weighted_avg: Whether to use weighted average considering segment length
            
        Returns:
            (doc_id, vector): Document ID and corresponding vector
        """
        # Split long document (with overlap)
        text_chunks = self.split_long_text_with_overlap(text, max_tokens=7600, overlap_ratio=overlap_ratio)
        
        chunk_lengths = [len(chunk) for chunk in text_chunks]
        chunk_tokens = [self.estimate_tokens(chunk) for chunk in text_chunks]
        
        print(f"Document {doc_id} split into {len(text_chunks)} segments with overlap")
        if len(text_chunks) > 1:
            print(f"Average segment length: {sum(chunk_lengths)/len(chunk_lengths):.1f} characters, average tokens: {sum(chunk_tokens)/len(chunk_tokens):.1f} tokens")
            print(f"Max segment: {max(chunk_tokens)} tokens, min segment: {min(chunk_tokens)} tokens")
            print(f"Overlap ratio: {overlap_ratio*100:.1f}%, using {'weighted' if use_weighted_avg else 'simple'} average")
        
        if len(text_chunks) == 1:
            # Document is short enough, process directly
            try:
                print(f"Starting to process document {doc_id}, length: {len(text_chunks[0])} characters, approx. {chunk_tokens[0]} tokens")
                embeddings = self._make_request(text_chunks[0], batch_size=1)
                if embeddings:
                    return doc_id, embeddings[0]
                else:
                    print(f"Warning: Failed to get embedding vector for document {doc_id}")
                    return None, None
            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")
                return None, None
        else:
            # Document is too long, process in segments and perform weighted average
            all_embeddings = []
            weights = []
            
            for i, chunk in enumerate(text_chunks):
                try:
                    est_tokens = chunk_tokens[i]
                    print(f"Processing document {doc_id} segment {i+1}/{len(text_chunks)}, length: {len(chunk)} characters, approx. {est_tokens} tokens")
                    embeddings = self._make_request(chunk, batch_size=1)
                    if embeddings:
                        all_embeddings.append(embeddings[0])
                        # Use text length or token count as weight
                        weights.append(est_tokens if use_weighted_avg else 1.0)
                        print(f"Segment {i+1} of document {doc_id} processed successfully")
                except Exception as e:
                    print(f"Error processing segment {i+1} of document {doc_id}: {e}")
                    
                # Add interval between requests to avoid API rate limiting
                if i < len(text_chunks) - 1:
                    print(f"Waiting 2 seconds before processing next segment...")
                    time.sleep(2)
            
            if not all_embeddings:
                print(f"Warning: All segments of document {doc_id} failed to get embedding vectors")
                return None, None
                
            # Calculate weighted average embedding vector
            print(f"Merging {len(all_embeddings)} segment vectors for document {doc_id} (using {'weighted' if use_weighted_avg else 'simple'} average)")
            
            if use_weighted_avg and sum(weights) > 0:
                # Normalize weights
                norm_weights = np.array(weights) / sum(weights)
                # Weighted average
                embeddings_array = np.array(all_embeddings)
                weighted_avg = np.zeros_like(embeddings_array[0])
                for i, embedding in enumerate(embeddings_array):
                    weighted_avg += embedding * norm_weights[i]
                avg_embedding = weighted_avg.tolist()
            else:
                # Simple average
                avg_embedding = np.mean(np.array(all_embeddings), axis=0).tolist()
                
            # Normalize final vector
            avg_embedding = np.array(avg_embedding)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = (avg_embedding / norm).tolist()
                
            return doc_id, avg_embedding
            
    def process_documents_enhanced(self, chunk_size=10, overlap_ratio=0.2, use_weighted_avg=True):
        """
        Enhanced document processing method, automatically handles long documents with overlap
        
        Args:
            chunk_size: Number of documents to process at a time
            overlap_ratio: Overlap ratio, default 20%
            use_weighted_avg: Whether to use weighted average considering segment length
            
        Returns:
            tuple: (doc_ids, vectors)
        """
        all_doc_ids = []
        all_vectors = []
        
        # Calculate total documents
        total_docs = len(self.doc_ids)
        print(f"Starting to process {total_docs} documents (overlap ratio: {overlap_ratio*100:.1f}%, using {'weighted' if use_weighted_avg else 'simple'} average)...")
        
        for i in range(0, total_docs, chunk_size):
            end_idx = min(i + chunk_size, total_docs)
            print(f"Processing documents {i+1} to {end_idx} (of {total_docs} total)...")
            
            for j in range(i, end_idx):
                doc_id = self.doc_ids[j]
                text = self.contexts[j]
                
                print(f"Processing document {j+1}/{total_docs}: {doc_id} (approx. {self.estimate_tokens(text)} tokens)")
                result_id, vector = self.process_document_with_overlap(doc_id, text, overlap_ratio, use_weighted_avg)
                
                if result_id is not None and vector is not None:
                    all_doc_ids.append(result_id)
                    all_vectors.append(vector)
            
            print(f"Completed processing {end_idx}/{total_docs} documents")
        
        # Normalize vectors
        if all_vectors:
            print("Performing vector normalization...")
            vectors = np.array(all_vectors)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            zero_mask = norms == 0
            norms[zero_mask] = 1.0  # Avoid division by zero
            vectors = vectors / norms
            all_vectors = vectors.tolist()
        
        print(f"Processing complete! Successfully generated {len(all_vectors)} document vectors")
        return all_doc_ids, all_vectors
    
class query_vectorizer:
    '''
    input: query str
    output: query_vector List[float32]
    '''
    def __init__(self, query: str, api_key: str, model_name: str):
        self.query = query
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def _make_request(self, text: str) -> List[float]:
        """
        Send synchronous request to SILICONFLOW API and get embedding vector for query text
        
        Args:
            text: Query text string
            
        Returns:
            Embedding vector
        """
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float"
        }
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                print(f"Sending query API request, attempt {retry_count+1}...")
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                # Extract embedding vector
                embedding = data['data'][0]['embedding']
                print(f"Query API request successful, received vector with dimension: {len(embedding)}")
                return embedding
                
            except requests.RequestException as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"Query API request failed after {retry_count} retries: {e}")
                    if hasattr(e, 'response') and e.response:
                        print(f"Response content: {e.response.text}")
                    raise
                else:
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))  # exponential backoff
                    print(f"Query request failed, waiting {wait_time} seconds before retrying ({retry_count}/{self.max_retries})...")
                    print(f"Error message: {str(e)}")
                    time.sleep(wait_time)

    def vectorize(self) -> List[float]:
        """
        Vectorize query text
        
        Returns:
            List[float]: Vector representation of query text
        """
        print(f"Starting to process query text: {self.query[:100]}...")
        
        # Get embedding vector
        vector = self._make_request(self.query)
        
        # L2 normalize vector
        print("Performing vector normalization...")
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        print("Query vectorization complete!")
        return vector.tolist()

'''
if __name__ == "__main__":
    doc_ids = ["1", "2", "3"]
    contexts = ["Hello, world!", "This is a test.", "Another example."]
    api_key = os.getenv("SILICON_API_KEY", "")
    model_name = "BAAI/bge-m3"

    vectorizer = doc_vectorizer(doc_ids, contexts, api_key, model_name)
    doc_ids, vectors = vectorizer.vectorize()
    print(doc_ids[0])
    print(vectors[0])
'''
