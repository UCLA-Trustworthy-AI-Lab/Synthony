# Database Persistence & Security Logging - Implementation v2.0

**Status**: ✅ IMPLEMENTED  
**Last Updated**: 2026-07-02  
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

# Opt-in: enables GET /sessions (session enumeration). Off by default --
# see "Known Limitations" below.
ENABLE_SESSION_LISTING=false
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

### Session Endpoints

Implemented in `src/synthony/api/endpoints.py`.

#### GET `/sessions`

List sessions with metadata. **Disabled by default** — returns `404` unless
`ENABLE_SESSION_LISTING=true`. Unlike the other session routes, this lets a
caller discover session_ids they don't already have, so it's opt-in (see
"Known Limitations").

Query params: `limit` (default 100, max 500), `offset` (default 0),
`include_expired` (default `false`).

```bash
curl "http://localhost:8000/sessions?limit=50"
```

```json
{
  "total": 2,
  "limit": 50,
  "offset": 0,
  "sessions": [
    {"session_id": "a3f2b1c0-...", "created_at": "2026-07-01T...", "expires_at": "2026-07-31T..."}
  ]
}
```

Note: `ip_address` is intentionally omitted from the list view even when
enabled (bulk IP exposure is a bigger privacy concern than a one-off lookup).

#### GET `/sessions/{session_id}`

Retrieve session details with all datasets and their analyses.

```bash
curl "http://localhost:8000/sessions/a3f2b1c0-..."
```

```json
{
  "session_id": "a3f2b1c0-...",
  "created_at": "2026-07-01T...",
  "expires_at": "2026-07-31T...",
  "ip_address": "127.0.0.1",
  "datasets": [
    {
      "dataset_id": "b4e3c2d1-...",
      "filename": "insurance.csv",
      "upload_status": "completed",
      "analyses": [{"analysis_id": "...", "status": "completed", "has_recommendation": false}]
    }
  ]
}
```

Returns `404` if the session doesn't exist.

#### GET `/sessions/{session_id}/data/{dataset_id}`

Download the original uploaded file.

```bash
curl -O "http://localhost:8000/sessions/a3f2b1c0-.../data/b4e3c2d1-..."
```

Returns `404` if the dataset doesn't exist, doesn't belong to that session,
or the file is missing on disk.

#### GET `/sessions/{session_id}/analyses/{analysis_id}`

Retrieve cached analysis results (dataset profile, column analysis, and
recommendation if one was generated), scoped to the given session.

```bash
curl "http://localhost:8000/sessions/a3f2b1c0-.../analyses/c5f4d3e2-..."
```

Returns `404` if the analysis doesn't exist or belongs to a different session.

#### DELETE `/sessions/{session_id}`

Delete a session and all associated data: uploaded files, dataset rows, and
analysis rows (cascade).

```bash
curl -X DELETE "http://localhost:8000/sessions/a3f2b1c0-..."
```

```json
{
  "session_id": "a3f2b1c0-...",
  "deleted": true,
  "files_deleted": 1,
  "bytes_freed": 12345,
  "datasets_deleted": 1,
  "analyses_deleted": 1,
  "message": "Session a3f2b1c0-... deleted: 1 files (12345 bytes), 1 datasets, 1 analyses removed"
}
```

Deletes files from disk first, then the database rows — see
`get_storage_manager().delete_session()` / `delete_session_by_id()` in
`database.py` for the retry-safety rationale. Returns `404` if the session
doesn't exist.

#### GET `/storage/stats`

Storage usage statistics, combining filesystem counts with a DB cross-check.

```bash
curl "http://localhost:8000/storage/stats"
```

```json
{
  "total_size_gb": 0.012,
  "storage_limit_gb": 10,
  "usage_percent": 0.12,
  "active_sessions": 3,
  "total_datasets": 5,
  "db_active_sessions": 3,
  "db_total_datasets": 5
}
```

`active_sessions`/`total_datasets` count session directories present on disk
(`storage.py::get_storage_stats`), regardless of whether the DB considers
them expired. `db_active_sessions` counts only non-expired sessions per
`count_sessions(include_expired=False)`. The two are expected to diverge on
a long-lived deployment: a large gap (filesystem count >> DB active count)
means expired sessions aren't being cleaned up from disk, which is exactly
what this cross-check is for — it's not a bug if they don't match.

---

## Known Limitations

- **No authentication.** No auth middleware exists anywhere in this API.
  `session_id` (an unguessable UUID) is the sole thing gating access to a
  dataset's contents — the same trust model `/analyze` and `/recommend`
  already rely on. Detail/delete/download-by-known-ID routes above are
  consistent with that; they don't introduce a new exposure.
- **`GET /sessions` is the one new capability**: nothing before this let a
  caller discover session_ids they don't already have. It's gated behind
  `ENABLE_SESSION_LISTING` (default off) and strips `ip_address` from its
  response for that reason.
- Recommended for internal/trusted-network deployments; not intended to be
  exposed to the public internet without an auth layer in front of it.

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

# Session listing/detail (backing the REST endpoints above)
sessions = list_sessions(include_expired=False, limit=100, offset=0)
total = count_sessions(include_expired=False)
detail = get_session_with_details(session_id)  # eager-loaded, safe to nest
analysis = get_analysis_with_relations(analysis_id)  # eager-loaded system_prompt
summary = delete_session_by_id(session_id)  # {datasets_deleted, analyses_deleted}
```

---

## UI Integration Workflow

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
# Check storage usage
curl http://localhost:8000/storage/stats

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
