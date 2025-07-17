from sqlalchemy import Column, String, Boolean, DateTime, Text, Enum, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext
import uuid
import enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from core.database import Base

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(enum.Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"

class User(Base):
    __tablename__ = "users"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False, index=True)
    role = Column(Enum(UserRole), default=UserRole.USER, index=True)
    
    # Profile Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    
    # User Metadata
    user_metadata = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Security Fields
    failed_login_attempts = Column(String(10), default="0")
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    files = relationship("File", foreign_keys="File.user_id", back_populates="user")
    upload_sessions = relationship("UploadSession", back_populates="user")
    annotations = relationship("Annotation", back_populates="user")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_role_active', 'role', 'is_active'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_last_login', 'last_login_at'),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role.value})>"
    
    @hybrid_property
    def full_name(self) -> str:
        """Get full name from first_name and last_name"""
        return f"{self.first_name} {self.last_name}".strip()

    @hybrid_property
    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return self.role in [UserRole.SUPER_ADMIN, UserRole.ADMIN]
    
    @hybrid_property
    def is_super_admin(self) -> bool:
        """Check if user is super admin"""
        return self.role == UserRole.SUPER_ADMIN
    
    @hybrid_property
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str) -> None:
        """Set user password with proper hashing"""
        self.hashed_password = pwd_context.hash(password)
        self.password_changed_at = datetime.utcnow()
    
    def increment_failed_login(self) -> None:
        """Increment failed login attempts and lock if necessary"""
        current_attempts = int(self.failed_login_attempts or "0")
        current_attempts += 1
        self.failed_login_attempts = str(current_attempts)
        
        # Lock account after 5 failed attempts for 30 minutes
        if current_attempts >= 5:
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)
    
    def reset_failed_login(self) -> None:
        """Reset failed login attempts on successful login"""
        self.failed_login_attempts = "0"
        self.locked_until = None
        self.last_login_at = datetime.utcnow()
    
    def update_last_login(self) -> None:
        """Update last login timestamp"""
        self.last_login_at = datetime.utcnow()
    
    def can_access_admin(self) -> bool:
        """Check if user can access admin features"""
        return self.is_active and self.is_verified and self.is_admin
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary representation"""
        data = {
            "id": str(self.id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "user_metadata": self.user_metadata,
        }
        
        if include_sensitive:
            data.update({
                "failed_login_attempts": self.failed_login_attempts,
                "locked_until": self.locked_until.isoformat() if self.locked_until else None,
                "password_changed_at": self.password_changed_at.isoformat() if self.password_changed_at else None,
            })
        
        return data

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    token = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Token Management
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_revoked = Column(Boolean, default=False, index=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # Support IPv6
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_refresh_token_user_active', 'user_id', 'is_revoked'),
        Index('idx_refresh_token_expires', 'expires_at'),
        Index('idx_refresh_token_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_valid(self) -> bool:
        """Check if token is valid (not expired and not revoked)"""
        return not self.is_expired and not self.is_revoked
    
    def revoke(self) -> None:
        """Revoke the refresh token"""
        self.is_revoked = True
    
    def update_last_used(self) -> None:
        """Update last used timestamp"""
        self.last_used_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert refresh token to dictionary representation"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "expires_at": self.expires_at.isoformat(),
            "is_revoked": self.is_revoked,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
        } 