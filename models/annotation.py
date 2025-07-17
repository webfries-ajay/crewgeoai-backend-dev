from sqlalchemy import Column, String, DateTime, Boolean, Float, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from core.database import Base

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Annotation details
    annotation_type = Column(String(50), nullable=False)  # 'defect_detection', 'bounding_box', 'polygon', 'point', 'measurement'
    coordinates = Column(JSON, nullable=False)  # Coordinate data
    properties = Column(JSON, default={})  # Additional properties like defect type, severity, etc.
    label = Column(String(255))
    
    # AI-generated metadata
    created_by_ai = Column(Boolean, default=False)
    ai_agent = Column(String(50))
    confidence_score = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    file = relationship("File", back_populates="annotations")
    user = relationship("User", back_populates="annotations") 