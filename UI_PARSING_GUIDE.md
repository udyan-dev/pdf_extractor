# PDF Extractor - UI Response Parsing Guide

## Overview

The PDF Extractor API returns a comprehensive, hierarchical JSON response that combines:
- **Layout data**: Raw blocks with geometry and text
- **Structure**: Hierarchy with parent-child relationships  
- **Timeline**: Extracted date ranges and events
- **Relationships**: Graph edges between blocks
- **Metadata**: Confidence scores, pipeline info, request tracking

This guide explains how to parse and render the response efficiently on the UI side.

---

## Response Structure

```json
{
  "meta": { ... },           // Request metadata, timing, LLM info
  "pages": [ ... ],          // Page-level statistics
  "blocks": [ ... ],         // Individual extracted blocks
  "hierarchy": [ ... ],      // Section tree with nesting
  "relationships": [ ... ],  // Block adjacency and containment edges
  "timeline": [ ... ],       // Extracted dates and events
  "confidence": { ... },     // Confidence metrics
  "validation": { ... },     // Validation status and issues
  "layout": { "blocks": [] },// Duplicate layout data (for backwards compat)
  "structure": { "sections": [] }, // Section metadata
  "raw_text": "..."          // Full reconstructed document text
}
```

---

## Parsing Hierarchy (Recommended)

### 1. Validate the Response

```javascript
const response = await fetch('/extract', { method: 'POST', body: formData });
const data = await response.json();

// Check validation and confidence
if (!data.validation?.passed) {
  console.warn('Validation failed:', data.validation.issues);
}

if (data.confidence?.overall < 0.7) {
  console.warn('Low confidence:', data.confidence.overall);
}

console.log('Request ID:', data.meta.request.request_id);
console.log('Job ID:', data.meta.request.job_id);
```

### 2. Build the Hierarchy View (Recommended Primary Path)

Use `hierarchy` for rendering the document structure:

```javascript
function renderHierarchy(node, depth = 0) {
  // Find block in blocks array
  const block = data.blocks.find(b => b.id === node.block_id);
  if (!block) return;

  // Render based on role
  const indent = '  '.repeat(depth);
  console.log(`${indent}[${block.role}] ${block.text.substring(0, 60)}`);
  console.log(`${indent}  Confidence: ${(block.confidence * 100).toFixed(1)}%`);

  // Recurse children
  if (node.children && node.children.length > 0) {
    node.children.forEach(child => renderHierarchy(child, depth + 1));
  }
}

// Entry point: render root hierarchy
data.hierarchy.forEach(node => renderHierarchy(node));
```

### 3. Enrich with Relationships

Add adjacency and containment info:

```javascript
function findRelationships(blockId) {
  return data.relationships.filter(
    rel => rel.source === blockId || rel.target === blockId
  );
}

// Example: Find all blocks that follow a given block
const nextBlocks = data.relationships
  .filter(rel => rel.type === 'next' && rel.source === 'b-1-0')
  .map(rel => data.blocks.find(b => b.id === rel.target));
```

### 4. Display Timeline (For Resumes/CVs)

Extract and render date-based events:

```javascript
// Timeline is already extracted in data.timeline
const events = data.timeline;

events.forEach(event => {
  console.log(`${event.date_span}`);
  console.log(`  Block: ${event.block_id}`);
  console.log(`  Context: ${event.context}`);
  console.log(`  Confidence: ${(event.confidence * 100).toFixed(1)}%`);
});
```

---

## Data Mapping Reference

### Block Object

```javascript
{
  "id": "b-1-0",             // Unique block identifier
  "page": 1,                 // Page number (1-indexed)
  "bbox": [x0, y0, x1, y1],  // Bounding box in PDF units
  "text": "...",             // Extracted text
  "role": "heading",         // "heading", "paragraph", "list", "table", "metadata"
  "level": 1,                // Nesting level (0 = no level)
  "column": 0,               // Column index (-1 = centered header)
  "font": 16.0,              // Font size
  "confidence": 0.898,       // Classifier confidence [0, 1]
  "source": "deterministic", // "deterministic" or "validated_llm"
  "evidence": [],            // LLM evidence spans (if source is validated_llm)
  "features": {              // Simplified feature flags
    "bullet": true,          // Has bullet points
    "table": false,          // Is a table
    "metadata": false        // Contains contact/metadata
  },
  "scores": {                // Role probability scores
    "heading": 0.911,
    "paragraph": 0.306,
    "list": 0.023
  }
}
```

### Hierarchy Node

```javascript
{
  "block_id": "b-1-0",
  "page": 1,
  "role": "heading",
  "level": 1,
  "children": [
    { "block_id": "b-1-1", ... },
    { "block_id": "b-1-2", ... }
  ]
}
```

### Timeline Event

```javascript
{
  "block_id": "b-1-4",
  "page": 1,
  "context": "Junior Software Engineer",    // First ~100 chars of block
  "date_span": "Aug 2024 - Present",       // Extracted date range
  "confidence": 0.9                         // Block confidence
}
```

### Relationship Edge

```javascript
{
  "type": "next",            // "next", "introduces", "belongs_to", "adjacent_table"
  "source": "b-1-0",
  "target": "b-1-1",
  "confidence": 0.92,
  "strategy": "reading_order" // "reading_order", "hierarchy", "graph"
}
```

---

## UI Rendering Patterns

### Resume/CV Layout

```javascript
function renderResume(data) {
  const name = data.blocks.find(b => b.id === data.hierarchy[0]?.block_id);
  const sections = data.hierarchy[0]?.children || [];

  // Header
  console.log(`# ${name?.text}`);

  // Contact info
  const contact = sections.find(s => data.blocks.find(b => b.id === s.block_id)?.role === 'metadata');
  if (contact) {
    const contactBlock = data.blocks.find(b => b.id === contact.block_id);
    console.log(`📧 ${contactBlock?.text}`);
  }

  // Sections
  sections.forEach(section => {
    const block = data.blocks.find(b => b.id === section.block_id);
    if (block?.role === 'heading') {
      console.log(`\n## ${block.text}`);
      section.children?.forEach(child => {
        const childBlock = data.blocks.find(b => b.id === child.block_id);
        if (childBlock?.role === 'list') {
          console.log(`- ${childBlock.text}`);
        } else if (childBlock?.role === 'metadata') {
          console.log(`**${childBlock.text}**`);
        }
      });
    }
  });
}
```

### Report/Article Layout

```javascript
function renderReport(data) {
  data.hierarchy.forEach(topSection => {
    const block = data.blocks.find(b => b.id === topSection.block_id);
    console.log(`## ${block.text}`);

    // Render body content in reading order
    const bodyRelationships = data.relationships.filter(
      r => r.source === topSection.block_id && r.type === 'next'
    );
    bodyRelationships.forEach(rel => {
      const bodyBlock = data.blocks.find(b => b.id === rel.target);
      console.log(`${bodyBlock.text}\n`);
    });
  });
}
```

---

## Performance Tips

### Lazy Load Large Responses

```javascript
// Use pagination for documents with >50 blocks
const blockBatches = chunkArray(data.blocks, 20);
blockBatches.forEach((batch, index) => {
  renderBatch(batch);
  if (index < blockBatches.length - 1) {
    scheduleNextRender(blockBatches[index + 1]);
  }
});

function chunkArray(arr, size) {
  const chunks = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}
```

### Cache Confidence Thresholds

```javascript
const MIN_CONFIDENCE = 0.7;

const highConfidenceBlocks = data.blocks.filter(
  b => b.confidence >= MIN_CONFIDENCE
);

const lowConfidenceBlocks = data.blocks.filter(
  b => b.confidence < MIN_CONFIDENCE
);
```

### Search in Extracted Data

```javascript
function findBlocksByRole(role) {
  return data.blocks.filter(b => b.role === role);
}

function findBlocksByText(query) {
  return data.blocks.filter(b =>
    b.text.toLowerCase().includes(query.toLowerCase())
  );
}

function findBlocksOnPage(pageNum) {
  return data.blocks.filter(b => b.page === pageNum);
}

const allHeadings = findBlocksByRole('heading');
const allBullets = findBlocksByRole('list');
const allMetadata = findBlocksByRole('metadata');
```

---

## Error Handling

```javascript
async function extractAndParse(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/extract', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json();
      
      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After');
        console.log(`Queue full. Retry after ${retryAfter}s`);
        // Implement exponential backoff
        await new Promise(r => setTimeout(r, parseInt(retryAfter) * 1000));
        return extractAndParse(file); // Retry
      } else if (response.status === 400) {
        console.error('Invalid PDF:', error.detail || error.error);
      } else if (response.status === 504) {
        console.error('Extraction timeout after', error.timeout_seconds, 'seconds');
      } else if (response.status === 413) {
        console.error('File too large. Max:', error.max_size_mb, 'MB');
      }
      return null;
    }

    const data = await response.json();

    // Validate structure
    if (!data.hierarchy || !data.blocks) {
      console.error('Invalid response structure');
      return null;
    }

    // Check validation
    if (!data.validation.passed) {
      console.warn('Validation issues:', data.validation.issues);
      // Continue anyway; some validation issues are non-fatal
    }

    // Check confidence
    if (data.confidence.overall < 0.65) {
      console.warn('Low overall confidence:', data.confidence.overall);
    }

    return data;
  } catch (err) {
    console.error('Extraction failed:', err.message);
    return null;
  }
}
```

---

## Request/Response Metadata

### Request Tracking

```javascript
// Track extraction job for support/debugging
console.log('Request ID:', data.meta.request.request_id);     // Unique request ID
console.log('Job ID:', data.meta.request.job_id);             // Internal job ID
console.log('Priority:', data.meta.request.priority);         // "small", "medium", "large"
console.log('Timeout:', data.meta.request.timeout_seconds);   // Timeout applied
console.log('Size:', data.meta.request.size_bytes / 1024, 'KB');
```

### Runtime Metrics

```javascript
const runtime = data.meta.runtime;
console.log(`Queue depth: ${runtime.queue_depth}`);
console.log(`Active jobs: ${runtime.active_jobs}`);
console.log(`Avg latency: ${runtime.avg_latency_seconds}s`);
console.log(`Completed: ${runtime.completed_jobs}, Failed: ${runtime.failed_jobs}`);
```

### LLM Usage

```javascript
const llm = data.meta.llm;
if (llm.enabled) {
  console.log(`LLM model: ${llm.model}`);
  if (llm.used) {
    console.log(`LLM refined ${llm.accepted_blocks.length} blocks`);
    llm.accepted_blocks.forEach(blockId => {
      const block = data.blocks.find(b => b.id === blockId);
      console.log(`  - ${block.text.substring(0, 40)}`);
    });
  }
}
```

### Pipeline Information

```javascript
const pipeline = data.meta.pipeline;
console.log(`Queue stage: ${pipeline.queue_stage}`);
console.log(`Parse stage: ${pipeline.parse_stage}`);
console.log(`Consensus stage: ${pipeline.consensus_stage}`);
console.log(`Validation stage: ${pipeline.validation_stage}`);
```

---

## Best Practices

1. **Always check `validation.passed`** before rendering
2. **Use `hierarchy` for structure**, not raw block order
3. **Filter blocks by `confidence`** if rendering for users (recommend >= 0.7)
4. **Cache relationships** by `(source, target)` for fast lookups
5. **Display `source` field** so users know if LLM was involved
6. **Show `timeline`** for resumes/CVs to help users validate extracted dates
7. **Implement retry logic** for 429 responses using `Retry-After` header
8. **Use `block.features`** as quick flags (bullet, table, metadata) for UI styling
9. **Parse `confidence` per block** to highlight uncertain extractions
10. **Keep raw `raw_text`** for fallback/copy-paste functionality

---

## Common Issues & Solutions

### Issue: Blocks appear in wrong order
**Solution**: Use `relationships` with `type: "next"` to traverse in correct reading order, not block array order

### Issue: Missing sections in hierarchy
**Solution**: Check `validation.issues` for `hierarchy_reference` errors; rebuild from scratch if corrupted

### Issue: Low confidence on resume dates
**Solution**: Check `timeline` for extracted dates; they have independent confidence scores

### Issue: LLM didn't refine expected blocks
**Solution**: Check `meta.llm.accepted_blocks` length; some blocks may not have met candidate criteria (line_count ≤8, token_count ≤120)

### Issue: Timeout on large files
**Solution**: Check `meta.request.timeout_seconds`; implement retry with exponential backoff; files >10MB may timeout

### Issue: 429 (Queue Full) errors
**Solution**: Implement backoff using `Retry-After` header; check `meta.runtime.queue_depth` before submitting new files
