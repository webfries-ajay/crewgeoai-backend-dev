from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Dict, Any
import logging
from datetime import timedelta

from core.database import get_db
from core.security import security, AuthenticationError
from models.admin import Admin, AdminRole, AdminRefreshToken
from schemas.admin import (
    AdminCreate,
    AdminUpdate,
    AdminResponse,
    AdminLogin,
    SuperAdminUpdate
)
from schemas.user import Token, TokenRefresh

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/auth", tags=["Admin Authentication"])
bearer_scheme = HTTPBearer()

def create_admin_response_data(admin: Admin) -> dict:
    """
    Helper function to create admin response data dictionary
    This avoids SQLAlchemy session issues when creating Pydantic models
    """
    return {
        "id": str(admin.id),
        "email": admin.email,
        "first_name": admin.first_name,
        "last_name": admin.last_name,
        "full_name": f"{admin.first_name} {admin.last_name}".strip(),
        "avatar_url": admin.avatar_url,
        "department": admin.department,
        "employee_id": admin.employee_id,
        "role": admin.role,
        "permissions": admin.permissions or [],
        "is_active": admin.is_active,
        "is_verified": admin.is_verified,
        "created_at": admin.created_at,
        "updated_at": admin.updated_at,
        "last_login_at": admin.last_login_at,
        "last_activity_at": admin.last_activity_at,
        "admin_metadata": admin.admin_metadata,
    }

# Dependency to get current admin from JWT token
async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> Admin:
    """
    Dependency to get current authenticated admin from JWT token
    """
    try:
        token = credentials.credentials
        admin = await security.get_current_admin(db, token)
        return admin
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency to get current active admin
async def get_current_active_admin(
    current_admin: Admin = Depends(get_current_admin)
) -> Admin:
    """
    Dependency to get current active admin
    """
    if not current_admin.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin account is deactivated"
        )
    return current_admin

# Dependency to require super admin privileges
async def require_super_admin(
    current_admin: Admin = Depends(get_current_active_admin)
) -> Admin:
    """
    Dependency to require super admin privileges
    """
    if not current_admin.is_super_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin privileges required"
        )
    return current_admin

@router.post("/register", response_model=AdminResponse, status_code=status.HTTP_201_CREATED)
async def register_admin(
    admin_data: AdminCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new admin account
    Note: In production, this should be protected by super admin authentication
    """
    try:
        # Check if admin already exists
        stmt = select(Admin).where(Admin.email == admin_data.email)
        result = await db.execute(stmt)
        existing_admin = result.scalar_one_or_none()
        
        if existing_admin:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )
        
        # Check if employee_id is unique (if provided)
        if admin_data.employee_id:
            stmt = select(Admin).where(Admin.employee_id == admin_data.employee_id)
            result = await db.execute(stmt)
            existing_employee = result.scalar_one_or_none()
            
            if existing_employee:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Employee ID already exists"
                )
        
        # Create new admin
        new_admin = Admin(
            email=admin_data.email,
            first_name=admin_data.first_name,
            last_name=admin_data.last_name,
            role=admin_data.role or AdminRole.SUPPORT_ADMIN,
            department=admin_data.department,
            employee_id=admin_data.employee_id,
            permissions=admin_data.permissions or [],
            is_verified=True  # Admins are auto-verified
        )
        new_admin.set_password(admin_data.password)
        
        db.add(new_admin)
        await db.commit()
        await db.refresh(new_admin)
        
        logger.info(f"New admin registered: {new_admin.email}")
        return AdminResponse(**create_admin_response_data(new_admin))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during admin registration: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin registration failed"
        )

@router.post("/login", response_model=Token)
async def login_admin(
    admin_credentials: AdminLogin,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate admin and return access and refresh tokens
    """
    try:
        # Authenticate admin using security manager
        admin = await security.authenticate_admin(
            db, admin_credentials.email, admin_credentials.password
        )
        
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=30)  # 30 minutes for admin
        if admin_credentials.remember_me:
            access_token_expires = timedelta(hours=8)  # 8 hours max for admin
        
        access_token = security.create_access_token(
            data={"sub": str(admin.id), "email": admin.email, "role": admin.role.value, "type": "admin"},
            expires_delta=access_token_expires
        )
        
        # Create refresh token
        user_agent = request.headers.get("user-agent")
        client_ip = request.client.host if request.client else None
        
        refresh_token_record = await security.create_admin_refresh_token_record(
            db, admin.id, user_agent, client_ip
        )
        
        # Update last login and activity
        admin.update_activity(client_ip)
        await db.commit()
        
        # Refresh admin to get updated data
        await db.refresh(admin)
        
        logger.info(f"Admin logged in: {admin.email}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token_record.token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            admin=create_admin_response_data(admin)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during admin login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=Token)
async def refresh_admin_token(
    token_data: TokenRefresh,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh admin access token using refresh token
    """
    try:
        # Verify refresh token using security manager
        refresh_token_record = await security.verify_admin_refresh_token(
            db, token_data.refresh_token
        )
        
        # Get admin
        admin = refresh_token_record.admin
        if not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is deactivated"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=security.access_token_expire_minutes)
        access_token = security.create_access_token(
            data={"sub": str(admin.id), "email": admin.email, "role": admin.role.value, "type": "admin"},
            expires_delta=access_token_expires
        )
        
        # Update last activity
        admin.update_activity()
        await db.commit()
        
        # Refresh admin to get updated data
        await db.refresh(admin)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token_record.token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            admin=create_admin_response_data(admin)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing admin token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout_admin(
    token_data: TokenRefresh,
    current_admin: Admin = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout admin and revoke refresh token
    """
    try:
        # Revoke the specific refresh token using security manager
        await security.revoke_admin_refresh_token(db, token_data.refresh_token)
        
        logger.info(f"Admin logged out: {current_admin.email}")
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Error during admin logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.post("/logout-all")
async def logout_all_admin(
    current_admin: Admin = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout admin from all devices by revoking all refresh tokens
    """
    try:
        revoked_count = await security.revoke_all_admin_tokens(db, current_admin.id)
        
        logger.info(f"Admin logged out from all devices: {current_admin.email} ({revoked_count} tokens revoked)")
        
        return {
            "message": f"Successfully logged out from all devices",
            "revoked_tokens": revoked_count
        }
        
    except Exception as e:
        logger.error(f"Error during admin logout all: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout all failed"
        )

@router.get("/me", response_model=AdminResponse)
async def get_current_admin_info(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get current admin information
    """
    return AdminResponse(**create_admin_response_data(current_admin))

@router.put("/me", response_model=AdminResponse)
async def update_current_admin(
    admin_update: AdminUpdate,
    current_admin: Admin = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current admin profile
    """
    try:
        # Update admin fields
        update_data = admin_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(current_admin, field, value)
        
        await db.commit()
        await db.refresh(current_admin)
        
        logger.info(f"Admin profile updated: {current_admin.email}")
        return AdminResponse(**create_admin_response_data(current_admin))
        
    except Exception as e:
        logger.error(f"Error updating admin profile: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        ) 