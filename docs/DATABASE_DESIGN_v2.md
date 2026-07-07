# Database Persistence & Security Logging - Implementation v2.0

**Status**: ✅ IMPLEMENTED  
**Last Updated**: 2026-01-16  
**Modules**: `database.py`, `storage.py`, `security.py`

---

## Overview

The Synthony API now includes **persistent storage** with session tracking and comprehensive security audit logging. Users can upload datasets that persist across sessions with full retrieval capabilities.

---

## Architecture

### Database: SQLite + SQLAlchemy

**Location**: `./data/synthony.db`  
**Implementation**: `src/synthony/api/database.py`

#### Tables

```sql
-- Session tracking
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW,
    ip_address TEXT,
    user_agent TEXT,
    expires_at TIMESTAMP  -- Auto-calculated: created_at + 30 days
);

-- System prompt versioning with hash-based deduplication
CREATE TABLE system_prompts (
    prompt_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- SHA256 hash for deduplication
    file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW,
    is_active BOOLEAN DEFAULT FALSE,
    UNIQUE(version, content_hash)  -- Prevent duplicate version+hash
);

-- Dataset metadata
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    format TEXT CHECK(format IN ('csv', 'parquet')),
    uploaded_at TIMESTAMP DEFAULT NOW,
    upload_status TEXT DEFAULT 'completed'
);

-- Cached analyses
CREATE TABLE analyses (
    analysis_id TEXT PRIMARY KEY,
    dataset_id TEXT REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    profile_json TEXT,
    column_analysis_json TEXT,
    recommendation_json TEXT,
    created_at TIMESTAMP DEFAULT NOW,
    status TEXT DEFAULT 'completed'
);

-- Security audit trail
CREATE TABLE audit_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    action TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW,
    ip_address TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    metadata_json TEXT
);
```

---

## File Storage

**Implementation**: `src/synthony/api/storage.py`

### Directory Structure

```
data/
├── synthony.db
├── uploads/
│   └── {session_id}/
│       ├── {dataset_id}.csv
│       └── {dataset_id}.parquet
└── logs/
    └── error.log
```

### Storage Quotas (Configurable)

- **Per File**: 100 MB
- **Per Session**: 500 MB  
- **Total**: 10 GB
- **Retention**: 30 days

### Configuration (`.env`)

```bash
DATABASE_URL=sqlite:///./data/synthony.db
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=100
MAX_SESSION_STORAGE_MB=500
MAX_TOTAL_STORAGE_GB=10
DATA_RETENTION_DAYS=30
AUDIT_LOG_ENABLED=true
```

---

## API Integration

### Modified Endpoints

#### POST `/analyze`

**Now Returns**: `session_id` and `analysis_id`

```json
{
  "session_id": "a3f2b1c0-...",
  "analysis_id": "b4e3c2d1-...",
  "dataset_id": "insurance",
  "dataset_profile": {...},
  "column_analysis": {...}
}
```

**Workflow**:

1. Create session with UUID
2. Save file to `./data/uploads/{session_id}/{dataset_id}.csv`
3. Store metadata in `datasets` table
4. Run analysis
5. Cache results in `analyses` table
6. Log to `audit_logs`

---

### Planned Endpoints (Not Implemented)

> **Status check (current codebase):** the database layer described above
> (`src/synthony/api/database.py`) is real and is used internally by
> `/analyze` and the `/systemprompt/*` routes. The REST endpoints listed
> below were planned as part of this design but were never wired into
> `src/synthony/api/endpoints.py` / `server.py` — there is currently no
> way to list, retrieve, or delete sessions over the API. Retrieval/cleanup
> is only possible by calling `database.py` functions directly (see
> "Database Operations" below) or via `cleanup_expired_sessions()`.

#### GET `/sessions`

List active sessions with metadata

#### GET `/sessions/{session_id}`

Retrieve session details with all datasets and analyses

#### GET `/sessions/{session_id}/data/{dataset_id}`

Download original uploaded file

#### GET `/sessions/{session_id}/analyses/{analysis_id}`

Retrieve cached analysis results

#### DELETE `/sessions/{session_id}`

Delete session and all associated data

#### GET `/storage/stats`

Get storage usage statistics

---

## Security & Audit Logging

**Implementation**: `src/synthony/api/security.py`

### Audit Trail

All operations logged to `audit_logs` table:

- Upload, analyze, recommend, download, delete
- IP address and user agent tracking
- Success/failure status
- Error messages

### Error Logging

**File**: `./logs/error.log`

All errors logged with:

- Session ID context
- Timestamp
- Action being performed
- Full error message
- Additional metadata

**Usage**:

```python
from synthony.api.security import log_error

error_msg = log_error(session_id, "analyze", exception, context={...})
```

---

## Database Operations

**Implementation**: `src/synthony/api/database.py`

### Key Functions

```python
# Initialize database
init_database()

# Create session
session = create_session(ip_address, user_agent, retention_days=30)

# Register dataset
dataset = create_dataset(session_id, filename, file_path, file_size, format)

# Cache analysis
analysis = create_analysis(dataset_id, profile_json, column_analysis_json)

# Audit logging
log_audit(session_id, action, endpoint, ip_address, success=True)

# Cleanup expired sessions
count = cleanup_expired_sessions()
```

---

## UI Integration Workflow

> **Note:** this workflow assumes the `/sessions/*` endpoints above are
> live. As of this writing they are not implemented — this section
> describes the intended frontend integration once they are.

### Frontend Upload Flow

**Step 1**: Upload File

```javascript
const response = await fetch('/analyze', {
  method: 'POST',
  body: formData
});

const { session_id, dataset_id, analysis_id } = await response.json();
```

**Step 2**: Check Upload Status

```javascript
const session = await fetch(`/sessions/${session_id}`).then(r => r.json());
const dataset = session.datasets.find(d => d.dataset_id === dataset_id);

if (dataset.upload_status === 'completed') {
  // Enable analyze button
  enableAnalyzeButton();
}
```

**Step 3**: Validate State

```javascript
// 1) Check dataset uploaded
// 2) Check analysis stored  
// 3) Retrieve JSON
// 4) Run recommender
```

**Error Handling**:

```javascript
try {
  await runAnalysis();
} catch (error) {
  // Error logged to ./logs/error.log
  displayError(error.message);
}
```

---

## Session Management

### Lifecycle

1. **Upload** → Create session with UUID
2. **Store** → Save to disk + database
3. **Analyze** → Cache results
4. **Retrieve** → Access anytime within 30 days
5. **Expire** → Auto-cleanup after retention period

### Automatic Cleanup

Expired sessions (>30 days) are automatically deleted:

- Database records removed
- Files deleted from storage
- Storage space freed

---

## Migration from Stateless

**Before** (v1.0):

- Files deleted immediately after analysis
- No persistence
- No session tracking

**After** (v2.0):

- Files persist for 30 days
- Full session management
- Audit trail
- Retrieval capabilities

**Backward Compatibility**: ✅ Maintained  
Old endpoints still work, new fields added to responses.

---

## Production Deployment

### Initialization

```bash
# Database auto-creates on first startup
./start_api.py
# ✓ Database initialized: sqlite:///./data/synthony.db
```

### Monitoring

```bash
# Check storage usage (endpoint not implemented; query the DB directly instead)
sqlite3 data/synthony.db "SELECT COUNT(*), SUM(file_size) FROM datasets"

# View audit logs  
sqlite3 data/synthony.db "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10"

# Check error log
tail -f logs/error.log
```

### Maintenance

```bash
# Cleanup expired sessions (automatic, but can trigger manually)
python -c "from synthony.api.database import cleanup_expired_sessions; print(f'Cleaned up {cleanup_expired_sessions()} sessions')"
```

---

## Future Enhancements

- [ ] PostgreSQL migration for production scale
- [ ] S3/cloud storage integration
- [ ] Data encryption at rest
- [ ] Multi-user authentication
- [ ] Role-based access control (RBAC)
- [ ] Real-time audit log streaming
- [ ] Vector embeddings for RAG
