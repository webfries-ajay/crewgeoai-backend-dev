from sqlalchemy import Column, String, Boolean, DateTime, Text, Enum, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext
import uuid
import enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from core.database import Base

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AdminRole(enum.Enum):
    """Admin roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"  # Full system access
    SYSTEM_ADMIN = "system_admin"  # System configuration
    USER_ADMIN = "user_admin"  # User management
    CONTENT_ADMIN = "content_admin"  # Content moderation
    ANALYTICS_ADMIN = "analytics_admin"  # Analytics and reports
    SUPPORT_ADMIN = "support_admin"  # Customer support

class AdminPermission(enum.Enum):
    """Granular admin permissions"""
    # User Management
    CREATE_USERS = "create_users"
    UPDATE_USERS = "update_users"
    DELETE_USERS = "delete_users"
    VIEW_USERS = "view_users"
    
    # Admin Management
    CREATE_ADMINS = "create_admins"
    UPDATE_ADMINS = "update_admins"
    DELETE_ADMINS = "delete_admins"
    VIEW_ADMINS = "view_admins"
    
    # System Management
    SYSTEM_CONFIG = "system_config"
    DATABASE_ACCESS = "database_access"
    SERVER_MANAGEMENT = "server_management"
    
    # Content Management
    MODERATE_CONTENT = "moderate_content"
    MANAGE_PROJECTS = "manage_projects"
    MANAGE_FILES = "manage_files"
    
    # Analytics & Reports
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"
    VIEW_LOGS = "view_logs"
    
    # Support
    VIEW_SUPPORT_TICKETS = "view_support_tickets"
    RESPOND_SUPPORT = "respond_support"

class Admin(Base):
    __tablename__ = "admins"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False, index=True)
    role = Column(Enum(AdminRole), default=AdminRole.SUPPORT_ADMIN, index=True)
    
    # Profile Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    avatar_url = Column(String(500), nullable=True)
    department = Column(String(100), nullable=True)
    employee_id = Column(String(50), unique=True, nullable=True, index=True)
    
    # Permissions (stored as JSON array)
    permissions = Column(JSONB, default=list)  # List of AdminPermission values
    
    # Admin Metadata
    admin_metadata = Column(JSONB, default=dict)
    
    # Security & Access
    last_login_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_login_ip = Column(String(45), nullable=True)  # Support IPv6
    failed_login_attempts = Column(String(10), default="0")
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Audit Fields
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by_admin_id = Column(UUID(as_uuid=True), ForeignKey("admins.id"), nullable=True)
    
    # Activity Tracking
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    session_timeout_minutes = Column(String(10), default="480")  # 8 hours default
    
    # Relationships
    created_by = relationship("Admin", remote_side=[id], back_populates="created_admins")
    created_admins = relationship("Admin", back_populates="created_by")
    admin_refresh_tokens = relationship("AdminRefreshToken", back_populates="admin", cascade="all, delete-orphan")
    admin_sessions = relationship("AdminSession", back_populates="admin", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_admin_email_active', 'email', 'is_active'),
        Index('idx_admin_role_active', 'role', 'is_active'),
        Index('idx_admin_created_at', 'created_at'),
        Index('idx_admin_last_login', 'last_login_at'),
        Index('idx_admin_employee_id', 'employee_id'),
    )
    
    def __repr__(self):
        return f"<Admin(id={self.id}, email={self.email}, role={self.role.value})>"
    
    @hybrid_property
    def full_name(self) -> str:
        """Get full name from first_name and last_name"""
        return f"{self.first_name} {self.last_name}".strip()
    
    @hybrid_property
    def is_super_admin(self) -> bool:
        """Check if admin is super admin"""
        return self.role == AdminRole.SUPER_ADMIN
    
    @hybrid_property
    def is_system_admin(self) -> bool:
        """Check if admin has system admin privileges"""
        return self.role in [AdminRole.SUPER_ADMIN, AdminRole.SYSTEM_ADMIN]
    
    @hybrid_property
    def is_locked(self) -> bool:
        """Check if admin account is locked"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    @hybrid_property
    def session_expired(self) -> bool:
        """Check if admin session has expired"""
        if self.last_activity_at is None:
            return True
        
        timeout_minutes = int(self.session_timeout_minutes or "480")
        expiry_time = self.last_activity_at + timedelta(minutes=timeout_minutes)
        return datetime.utcnow() > expiry_time
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str) -> None:
        """Set admin password with proper hashing"""
        self.hashed_password = pwd_context.hash(password)
        self.password_changed_at = datetime.utcnow()
    
    def increment_failed_login(self) -> None:
        """Increment failed login attempts and lock if necessary"""
        current_attempts = int(self.failed_login_attempts or "0")
        current_attempts += 1
        self.failed_login_attempts = str(current_attempts)
        
        # Lock account after 3 failed attempts for 1 hour (stricter for admins)
        if current_attempts >= 3:
            self.locked_until = datetime.utcnow() + timedelta(hours=1)
    
    def reset_failed_login(self) -> None:
        """Reset failed login attempts on successful login"""
        self.failed_login_attempts = "0"
        self.locked_until = None
        self.last_login_at = datetime.utcnow()
    
    def update_activity(self, ip_address: str = None) -> None:
        """Update last activity and login info"""
        self.last_activity_at = datetime.utcnow()
        if ip_address:
            self.last_login_ip = ip_address
    
    def has_permission(self, permission: AdminPermission) -> bool:
        """Check if admin has specific permission"""
        if self.is_super_admin:
            return True
        
        # Check explicit permissions
        if permission.value in (self.permissions or []):
            return True
        
        # Check role-based permissions
        role_permissions = self.get_role_permissions()
        return permission.value in role_permissions
    
    def get_role_permissions(self) -> List[str]:
        """Get permissions based on admin role"""
        role_permission_map = {
            AdminRole.SUPER_ADMIN: [p.value for p in AdminPermission],  # All permissions
            AdminRole.SYSTEM_ADMIN: [
                AdminPermission.SYSTEM_CONFIG.value,
                AdminPermission.DATABASE_ACCESS.value,
                AdminPermission.SERVER_MANAGEMENT.value,
                AdminPermission.VIEW_ANALYTICS.value,
                AdminPermission.VIEW_LOGS.value,
                AdminPermission.VIEW_USERS.value,
                AdminPermission.VIEW_ADMINS.value,
            ],
            AdminRole.USER_ADMIN: [
                AdminPermission.CREATE_USERS.value,
                AdminPermission.UPDATE_USERS.value,
                AdminPermission.DELETE_USERS.value,
                AdminPermission.VIEW_USERS.value,
                AdminPermission.VIEW_ANALYTICS.value,
            ],
            AdminRole.CONTENT_ADMIN: [
                AdminPermission.MODERATE_CONTENT.value,
                AdminPermission.MANAGE_PROJECTS.value,
                AdminPermission.MANAGE_FILES.value,
                AdminPermission.VIEW_USERS.value,
            ],
            AdminRole.ANALYTICS_ADMIN: [
                AdminPermission.VIEW_ANALYTICS.value,
                AdminPermission.EXPORT_DATA.value,
                AdminPermission.VIEW_LOGS.value,
                AdminPermission.VIEW_USERS.value,
            ],
            AdminRole.SUPPORT_ADMIN: [
                AdminPermission.VIEW_SUPPORT_TICKETS.value,
                AdminPermission.RESPOND_SUPPORT.value,
                AdminPermission.VIEW_USERS.value,
            ],
        }
        
        return role_permission_map.get(self.role, [])
    
    def add_permission(self, permission: AdminPermission) -> None:
        """Add a specific permission to admin"""
        current_permissions = self.permissions or []
        if permission.value not in current_permissions:
            current_permissions.append(permission.value)
            self.permissions = current_permissions
    
    def remove_permission(self, permission: AdminPermission) -> None:
        """Remove a specific permission from admin"""
        current_permissions = self.permissions or []
        if permission.value in current_permissions:
            current_permissions.remove(permission.value)
            self.permissions = current_permissions
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert admin to dictionary representation"""
        data = {
            "id": str(self.id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "department": self.department,
            "employee_id": self.employee_id,
            "role": self.role.value,
            "permissions": self.permissions or [],
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
            "admin_metadata": self.admin_metadata,
        }
        
        if include_sensitive:
            data.update({
                "failed_login_attempts": self.failed_login_attempts,
                "locked_until": self.locked_until.isoformat() if self.locked_until else None,
                "password_changed_at": self.password_changed_at.isoformat() if self.password_changed_at else None,
                "last_login_ip": self.last_login_ip,
                "session_timeout_minutes": self.session_timeout_minutes,
            })
        
        return data

class AdminRefreshToken(Base):
    __tablename__ = "admin_refresh_tokens"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    token = Column(String(255), unique=True, index=True, nullable=False)
    admin_id = Column(UUID(as_uuid=True), ForeignKey("admins.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Token Management
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_revoked = Column(Boolean, default=False, index=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Relationships
    admin = relationship("Admin", back_populates="admin_refresh_tokens")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_admin_refresh_token_admin_active', 'admin_id', 'is_revoked'),
        Index('idx_admin_refresh_token_expires', 'expires_at'),
    )
    
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

class AdminSession(Base):
    __tablename__ = "admin_sessions"
    
    # Primary Fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    admin_id = Column(UUID(as_uuid=True), ForeignKey("admins.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Session Management
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # Session Metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    device_fingerprint = Column(String(255), nullable=True)
    
    # Relationships
    admin = relationship("Admin", back_populates="admin_sessions")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_admin_session_admin_active', 'admin_id', 'is_active'),
        Index('idx_admin_session_expires', 'expires_at'),
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    def extend_session(self, minutes: int = 480) -> None:
        """Extend session expiration"""
        self.expires_at = datetime.utcnow() + timedelta(minutes=minutes)
        self.last_activity_at = datetime.utcnow()
    
    def terminate(self) -> None:
        """Terminate the session"""
        self.is_active = False 