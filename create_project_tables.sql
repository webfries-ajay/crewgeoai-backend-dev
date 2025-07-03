-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    settings JSONB DEFAULT '{}',
    project_metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    location VARCHAR(255),
    total_file_size BIGINT NOT NULL DEFAULT 0,
    file_count INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    video_count INTEGER NOT NULL DEFAULT 0,
    document_count INTEGER NOT NULL DEFAULT 0,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion_at TIMESTAMP WITH TIME ZONE,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    collaborator_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    archived_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for projects
CREATE INDEX IF NOT EXISTS idx_project_owner_status ON projects(owner_id, status);
CREATE INDEX IF NOT EXISTS idx_project_category ON projects(category);
CREATE INDEX IF NOT EXISTS idx_project_created_at ON projects(created_at);
CREATE INDEX IF NOT EXISTS idx_project_updated_at ON projects(updated_at);
CREATE INDEX IF NOT EXISTS ix_projects_id ON projects(id);
CREATE INDEX IF NOT EXISTS ix_projects_owner_id ON projects(owner_id);
CREATE INDEX IF NOT EXISTS ix_projects_status ON projects(status);

-- Create project_collaborators table
CREATE TABLE IF NOT EXISTS project_collaborators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL DEFAULT 'viewer',
    permissions JSONB DEFAULT '[]',
    invited_by UUID REFERENCES users(id),
    invited_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    joined_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(project_id, user_id)
);

-- Create indexes for project_collaborators
CREATE INDEX IF NOT EXISTS idx_project_collaborator_project ON project_collaborators(project_id);
CREATE INDEX IF NOT EXISTS idx_project_collaborator_user ON project_collaborators(user_id);
CREATE INDEX IF NOT EXISTS ix_project_collaborators_id ON project_collaborators(id);

-- Create files table
CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    checksum_md5 VARCHAR(32),
    checksum_sha256 VARCHAR(64),
    virus_scan_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    virus_scan_result JSONB DEFAULT '{}',
    upload_status VARCHAR(20) NOT NULL DEFAULT 'uploading',
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    file_metadata JSONB DEFAULT '{}',
    exif_data JSONB DEFAULT '{}',
    geospatial_data JSONB DEFAULT '{}',
    width INTEGER,
    height INTEGER,
    duration_seconds INTEGER,
    frame_rate INTEGER,
    thumbnail_generated BOOLEAN NOT NULL DEFAULT FALSE,
    preview_generated BOOLEAN NOT NULL DEFAULT FALSE,
    analysis_completed BOOLEAN NOT NULL DEFAULT FALSE,
    uploaded_by UUID NOT NULL REFERENCES users(id),
    upload_session_id VARCHAR(255),
    tags JSONB DEFAULT '[]',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    last_accessed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for files
CREATE INDEX IF NOT EXISTS idx_file_project_type ON files(project_id, file_type);
CREATE INDEX IF NOT EXISTS idx_file_upload_status ON files(upload_status);
CREATE INDEX IF NOT EXISTS idx_file_processing_status ON files(processing_status);
CREATE INDEX IF NOT EXISTS idx_file_uploader ON files(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_file_created_at ON files(created_at);
CREATE INDEX IF NOT EXISTS ix_files_id ON files(id);
CREATE INDEX IF NOT EXISTS ix_files_project_id ON files(project_id);
CREATE INDEX IF NOT EXISTS ix_files_file_type ON files(file_type);

-- Create file_thumbnails table
CREATE TABLE IF NOT EXISTS file_thumbnails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    thumbnail_type VARCHAR(20) NOT NULL,
    thumbnail_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for file_thumbnails
CREATE INDEX IF NOT EXISTS idx_file_thumbnail_file_type ON file_thumbnails(file_id, thumbnail_type);
CREATE INDEX IF NOT EXISTS ix_file_thumbnails_id ON file_thumbnails(id);
CREATE INDEX IF NOT EXISTS ix_file_thumbnails_file_id ON file_thumbnails(file_id); 