CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    risk_level INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    has_audio BOOLEAN DEFAULT FALSE,
    has_video BOOLEAN DEFAULT FALSE,
    audio_url TEXT,
    video_url TEXT,
    model_version VARCHAR(128),
    inference_time_ms INTEGER,
    confidence DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

CREATE TABLE IF NOT EXISTS annotation_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    priority INTEGER DEFAULT 5,
    status VARCHAR(16) DEFAULT 'pending',
    ground_truth_risk INTEGER,
    ground_truth_empathy INTEGER,
    assigned_to VARCHAR(64),
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_annotation_tasks_status ON annotation_tasks(status, priority);
CREATE UNIQUE INDEX IF NOT EXISTS uq_annotation_tasks_message_id ON annotation_tasks(message_id);

CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(32) NOT NULL UNIQUE,
    model_path VARCHAR(256),
    data_version VARCHAR(32),
    metrics JSONB DEFAULT '{}'::jsonb,
    is_production BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ab_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(64),
    control_model VARCHAR(32),
    treatment_model VARCHAR(32),
    traffic_percent INTEGER DEFAULT 10,
    status VARCHAR(16) DEFAULT 'running',
    metrics JSONB DEFAULT '{}'::jsonb
);

INSERT INTO model_versions (version, model_path, data_version, metrics, is_production, deployed_at)
VALUES (
    'v1.0',
    'services/mock',
    'v1.0-data',
    '{"accuracy": 0.85, "f1": 0.82}'::jsonb,
    TRUE,
    NOW()
)
ON CONFLICT (version) DO NOTHING;
