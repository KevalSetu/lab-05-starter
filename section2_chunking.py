# section2_chunking.py
sample_documents = [
    """Customer Support Best Practices - Chapter 1
    
    When handling refund requests, always verify the purchase date first. Our policy allows 
    refunds within 30 days of purchase for most items. Electronic goods have a 14-day window. 
    Items on sale or clearance are final sale and cannot be refunded.
    
    Process:
    1. Check purchase date in the order system
    2. Verify item eligibility (not clearance, within window)
    3. Ask for reason (helps product team improve)
    4. Process refund to original payment method
    5. Send confirmation email within 24 hours
    
    Common issues: Customers often don't have receipts. You can look up orders by email 
    and order number. If unable to verify, escalate to a supervisor.
    """,
    
    """Customer Support Best Practices - Chapter 2
    
    Shipping inquiries are the most common support ticket type. Most shipping delays are 
    carrier-related and outside our control, but customers don't care whose fault it is.
    
    Response template for delayed shipments:
    "I understand how frustrating delays are. Let me check your order status... 
    [check tracking]... I can see it's currently in transit. Expected delivery is [date]. 
    If it doesn't arrive by then, contact us and we'll send a replacement immediately."
    
    For lost packages (tracking shows delivered but customer denies receipt):
    1. Check if it was signed for (if yes, verify signature)
    2. Ask customer to check with neighbours and building concierge
    3. File carrier claim if still missing after 48 hours
    4. Send replacement immediately (don't wait for claim resolution)
    """,
]

# Strategy 1: Fixed-size chunking (simple but crude)
def chunk_fixed_size(text, chunk_size=200):
    """Split text every N characters"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# Strategy 2: Paragraph-based chunking (respects structure)
def chunk_by_paragraph(text):
    """Split on double newlines (paragraph boundaries)"""
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]

# Strategy 3: Sentence-based with overlap (more sophisticated)
def chunk_by_sentences(text, sentences_per_chunk=3, overlap=1):
    """Split into sentence groups with overlap"""
    # Simple sentence splitting (would use spaCy in production)
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk = ' '.join(sentences[i:i+sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks

# Test all three strategies
for doc_idx, doc in enumerate(sample_documents):
    print(f"\nDocument {doc_idx + 1}:")
    print(f"Original length: {len(doc)} characters\n")
    
    fixed = chunk_fixed_size(doc, chunk_size=200)
    print(f"Fixed-size chunks (200 chars): {len(fixed)} chunks")
    print(f"Sample: {fixed[0][:100]}...\n")
    
    para = chunk_by_paragraph(doc)
    print(f"Paragraph chunks: {len(para)} chunks")
    print(f"Sample: {para[0][:100]}...\n")
    
    sent = chunk_by_sentences(doc, sentences_per_chunk=3, overlap=1)
    print(f"Sentence chunks (3 sent, 1 overlap): {len(sent)} chunks")
    print(f"Sample: {sent[0][:100]}...\n")
