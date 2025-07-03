from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import re

from models.admin import AdminRole, AdminPermission

class AdminCreate(BaseModel):
    """Schema for admin creation"""
    email: EmailStr = Field(..., description="Admin email address")
    password: str = Field(..., min_length=8, max_length=100, description="Admin password")
    first_name: str = Field(..., min_length=1, max_length=100, description="Admin first name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Admin last name")
    role: AdminRole = Field(AdminRole.SUPPORT_ADMIN, description="Admin role")
    department: Optional[str] = Field(None, max_length=100, description="Department")
    employee_id: Optional[str] = Field(None, max_length=50, description="Employee ID")
    permissions: Optional[List[AdminPermission]] = Field([], description="Additional permissions")
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('employee_id')
    def validate_employee_id(cls, v):
        if v and not re.match(r'^[A-Z0-9-]+$', v):
            raise ValueError('Employee ID can only contain uppercase letters, numbers, and hyphens')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "admin@crewgeoai.com",
                "password": "AdminPass123!",
                "first_name": "John",
                "last_name": "Smith",
                "role": "user_admin",
                "department": "IT",
                "employee_id": "EMP-001",
                "permissions": ["view_analytics"]
            }
        }

class AdminUpdate(BaseModel):
    """Schema for admin updates"""
    email: Optional[EmailStr] = Field(None, description="Admin email address")
    first_name: Optional[str] = Field(None, min_length=1, max_length=100, description="Admin first name")
    last_name: Optional[str] = Field(None, min_length=1, max_length=100, description="Admin last name")
    avatar_url: Optional[str] = Field(None, max_length=500, description="Avatar URL")
    department: Optional[str] = Field(None, max_length=100, description="Department")
    employee_id: Optional[str] = Field(None, max_length=50, description="Employee ID")
    admin_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional admin metadata")
    session_timeout_minutes: Optional[int] = Field(None, ge=30, le=1440, description="Session timeout in minutes")
    
    @validator('employee_id')
    def validate_employee_id(cls, v):
        if v and not re.match(r'^[A-Z0-9-]+$', v):
            raise ValueError('Employee ID can only contain uppercase letters, numbers, and hyphens')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "A. Smith",
                "department": "IT Security",
                "avatar_url": "https://example.com/avatar.jpg",
                "session_timeout_minutes": 480
            }
        }

class AdminResponse(BaseModel):
    """Schema for admin response"""
    id: str
    email: EmailStr
    first_name: str
    last_name: str
    full_name: str
    avatar_url: Optional[str] = None
    department: Optional[str] = None
    employee_id: Optional[str] = None
    role: AdminRole
    permissions: List[str] = []
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    admin_metadata: Optional[Dict[str, Any]] = None
    
    @validator('id', pre=True)
    def convert_id_to_string(cls, v):
        return str(v) if v is not None else None
    
    @validator('full_name', pre=True, always=True)
    def ensure_full_name(cls, v, values):
        if v:
            return v
        # If full_name is not provided, construct it from first_name and last_name
        first_name = values.get('first_name', '')
        last_name = values.get('last_name', '')
        return f"{first_name} {last_name}".strip()
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "admin@crewgeoai.com",
                "first_name": "John",
                "last_name": "Smith",
                "full_name": "John Smith",
                "role": "user_admin",
                "department": "IT",
                "employee_id": "EMP-001",
                "is_active": True,
                "is_verified": True,
                "permissions": ["view_analytics"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }

class AdminLogin(BaseModel):
    """Schema for admin login"""
    email: EmailStr = Field(..., description="Admin email address")
    password: str = Field(..., min_length=1, description="Admin password")
    remember_me: bool = Field(False, description="Remember admin login")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "admin@crewgeoai.com",
                "password": "AdminPass123!",
                "remember_me": False
            }
        }

class AdminToken(BaseModel):
    """Schema for admin JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration time in seconds")
    admin: AdminResponse = Field(..., description="Admin information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "secure_refresh_token_here",
                "token_type": "bearer",
                "expires_in": 1800,
                "admin": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "email": "admin@crewgeoai.com",
                    "role": "user_admin"
                }
            }
        }

class AdminPasswordChange(BaseModel):
    """Schema for admin password change"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "OldAdminPass123!",
                "new_password": "NewAdminPass456!"
            }
        }

class SuperAdminUpdate(AdminUpdate):
    """Schema for super admin updates with additional privileges"""
    role: Optional[AdminRole] = Field(None, description="Admin role")
    is_active: Optional[bool] = Field(None, description="Admin active status")
    is_verified: Optional[bool] = Field(None, description="Admin verification status")
    permissions: Optional[List[AdminPermission]] = Field(None, description="Admin permissions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "system_admin",
                "is_active": True,
                "is_verified": True,
                "permissions": ["system_config", "view_analytics"],
                "first_name": "System",
                "last_name": "Administrator"
            }
        }

class AdminPermissionUpdate(BaseModel):
    """Schema for updating admin permissions"""
    permissions_to_add: Optional[List[AdminPermission]] = Field([], description="Permissions to add")
    permissions_to_remove: Optional[List[AdminPermission]] = Field([], description="Permissions to remove")
    
    class Config:
        json_schema_extra = {
            "example": {
                "permissions_to_add": ["view_analytics", "export_data"],
                "permissions_to_remove": ["delete_users"]
            }
        }

class AdminSessionInfo(BaseModel):
    """Schema for admin session information"""
    session_id: str
    created_at: datetime
    last_activity_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_current: bool = False
    
    class Config:
        from_attributes = True

class AdminActivityLog(BaseModel):
    """Schema for admin activity logging"""
    admin_id: str
    action: str
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "admin_id": "123e4567-e89b-12d3-a456-426614174000",
                "action": "user_created",
                "resource": "user",
                "resource_id": "456e7890-e89b-12d3-a456-426614174000",
                "details": {"user_email": "newuser@example.com"},
                "ip_address": "192.168.1.100",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        } 