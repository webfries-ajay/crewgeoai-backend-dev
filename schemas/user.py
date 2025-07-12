from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import re

from models.user import UserRole

class UserCreate(BaseModel):
    """Schema for user creation"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=100, description="User password")
    first_name: str = Field(..., min_length=1, max_length=100, description="User first name")
    last_name: str = Field(..., min_length=1, max_length=100, description="User last name")
    role: Optional[UserRole] = Field(UserRole.USER, description="User role")
    
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "first_name": "John",
                "last_name": "Doe",
                "role": "user"
            }
        }

class UserUpdate(BaseModel):
    """Schema for user updates"""
    email: Optional[EmailStr] = Field(None, description="User email address")
    first_name: Optional[str] = Field(None, min_length=1, max_length=100, description="User first name")
    last_name: Optional[str] = Field(None, min_length=1, max_length=100, description="User last name")
    avatar_url: Optional[str] = Field(None, max_length=500, description="Avatar URL")
    bio: Optional[str] = Field(None, max_length=1000, description="User biography")
    user_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional user metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Smith",
                "bio": "Software developer passionate about AI and geospatial technology",
                "avatar_url": "https://example.com/avatar.jpg"
            }
        }

class UserResponse(BaseModel):
    """Schema for user response"""
    id: str
    email: EmailStr
    first_name: str
    last_name: str
    full_name: str
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    user_metadata: Optional[Dict[str, Any]] = None
    
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
                "email": "user@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "full_name": "John Doe",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }

class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")
    remember_me: bool = Field(False, description="Remember user login")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "remember_me": False
            }
        }

class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration time in seconds")
    user: Optional[UserResponse] = Field(None, description="User information")
    admin: Optional[Dict[str, Any]] = Field(None, description="Admin information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
                "user": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "email": "user@example.com",
                    "role": "user"
                }
            }
        }

class TokenRefresh(BaseModel):
    """Schema for token refresh"""
    refresh_token: str = Field(..., description="Refresh token")
    
    class Config:
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }

class PasswordChange(BaseModel):
    """Schema for password change"""
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
                "current_password": "OldPass123!",
                "new_password": "NewSecurePass456!"
            }
        }

class PasswordReset(BaseModel):
    """Schema for password reset request"""
    email: EmailStr = Field(..., description="User email address")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }

class ForgotPasswordRequest(BaseModel):
    """Schema for forgot password request"""
    email: EmailStr = Field(..., description="User email address")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }

class ResetPasswordRequest(BaseModel):
    """Schema for reset password request"""
    email: EmailStr = Field(..., description="User email address")
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
                "email": "user@example.com",
                "new_password": "NewSecurePass456!"
            }
        }

class AdminUserUpdate(UserUpdate):
    """Schema for admin user updates with additional privileges"""
    role: Optional[UserRole] = Field(None, description="User role")
    is_active: Optional[bool] = Field(None, description="User active status")
    is_verified: Optional[bool] = Field(None, description="User verification status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "admin",
                "is_active": True,
                "is_verified": True,
                "first_name": "Admin",
                "last_name": "User"
            }
        } 