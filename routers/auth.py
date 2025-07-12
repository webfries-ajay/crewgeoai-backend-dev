from fastapi import APIRouter, Depends, HTTPException, status, Request, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import logging
from datetime import timedelta

from core.database import get_db
from core.security import security, AuthenticationError
from models.user import User, UserRole, RefreshToken
from schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserLogin,
    Token,
    TokenRefresh,
    PasswordChange,
    PasswordReset,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    AdminUserUpdate
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])
bearer_scheme = HTTPBearer()

def create_user_response_data(user: User) -> dict:
    """
    Helper function to create user response data dictionary
    This avoids SQLAlchemy session issues when creating Pydantic models
    """
    return {
        "id": str(user.id),
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": f"{user.first_name} {user.last_name}".strip(),
        "avatar_url": user.avatar_url,
        "bio": user.bio,
        "role": user.role,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_login_at": user.last_login_at,
        "user_metadata": user.user_metadata,
    }

# Dependency to get current user from JWT token
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to get current authenticated user from JWT token
    """
    try:
        token = credentials.credentials
        user = await security.get_current_user(db, token)
        return user
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency to get current active user
async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active user
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    return current_user

# Dependency to require admin privileges
async def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Dependency to require admin privileges
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Dependency to require super admin privileges
async def require_super_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Dependency to require super admin privileges
    """
    if not current_user.is_super_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin privileges required"
        )
    return current_user

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account
    """
    try:
        # Check if user already exists
        stmt = select(User).where(User.email == user_data.email)
        result = await db.execute(stmt)
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )
        
        # Create new user
        new_user = User(
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role or UserRole.USER
        )
        new_user.set_password(user_data.password)
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        logger.info(f"New user registered: {new_user.email}")
        return UserResponse(**create_user_response_data(new_user))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during user registration: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=Token)
async def login_user(
    user_credentials: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return access and refresh tokens
    """
    try:
        # Authenticate user
        user = await security.authenticate_user(
            db, user_credentials.email, user_credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=security.access_token_expire_minutes)
        if user_credentials.remember_me:
            # Extend token life for "remember me"
            access_token_expires = timedelta(hours=24)
        
        access_token = security.create_access_token(
            data={"sub": str(user.id), "email": user.email, "role": user.role.value},
            expires_delta=access_token_expires
        )
        
        # Create refresh token
        user_agent = request.headers.get("user-agent")
        client_ip = request.client.host if request.client else None
        
        refresh_token_record = await security.create_refresh_token_record(
            db, user.id, user_agent, client_ip
        )
        
        # Update last login
        user.update_last_login()
        await db.commit()
        
        # Refresh user to get updated data
        await db.refresh(user)
        
        logger.info(f"User logged in: {user.email}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token_record.token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            user=UserResponse(**create_user_response_data(user))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    """
    try:
        # Verify refresh token
        refresh_token_record = await security.verify_refresh_token(
            db, token_data.refresh_token
        )
        
        # Get user
        stmt = select(User).where(User.id == refresh_token_record.user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or deactivated"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=security.access_token_expire_minutes)
        access_token = security.create_access_token(
            data={"sub": str(user.id), "email": user.email, "role": user.role.value},
            expires_delta=access_token_expires
        )
        
        # Create new refresh token and revoke old one
        refresh_token_record.revoke()
        
        user_agent = request.headers.get("user-agent")
        client_ip = request.client.host if request.client else None
        
        new_refresh_token_record = await security.create_refresh_token_record(
            db, user.id, user_agent, client_ip
        )
        
        await db.commit()
        
        # Refresh user to get latest data
        await db.refresh(user)
        
        logger.info(f"Token refreshed for user: {user.email}")
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token_record.token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            user=UserResponse(**create_user_response_data(user))
        )
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during token refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout_user(
    token_data: TokenRefresh,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user by revoking refresh token
    """
    try:
        # Revoke the refresh token
        success = await security.revoke_refresh_token(db, token_data.refresh_token)
        
        if success:
            logger.info(f"User logged out: {current_user.email}")
            return {"message": "Successfully logged out"}
        else:
            return {"message": "Token already revoked or invalid"}
            
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.post("/logout-all")
async def logout_all_devices(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user from all devices by revoking all refresh tokens
    """
    try:
        revoked_count = await security.revoke_all_user_tokens(db, current_user.id)
        
        logger.info(f"User logged out from all devices: {current_user.email} ({revoked_count} tokens revoked)")
        
        return {
            "message": f"Successfully logged out from all devices",
            "revoked_tokens": revoked_count
        }
        
    except Exception as e:
        logger.error(f"Error during logout all: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout all failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information
    """
    return UserResponse(**create_user_response_data(current_user))

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user profile
    """
    try:
        # Check for email conflicts
        if user_update.email:
            stmt = select(User).where(
                User.id != current_user.id,
                User.email == user_update.email
            )
            result = await db.execute(stmt)
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already in use"
                )
        
        # Update user fields
        update_data = user_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(current_user, field, value)
        
        await db.commit()
        await db.refresh(current_user)
        
        logger.info(f"User profile updated: {current_user.email}")
        return UserResponse(**create_user_response_data(current_user))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change user password
    """
    try:
        # Verify current password
        if not current_user.verify_password(password_data.current_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Set new password
        current_user.set_password(password_data.new_password)
        await db.commit()
        
        # Revoke all existing refresh tokens for security
        await security.revoke_all_user_tokens(db, current_user.id)
        
        logger.info(f"Password changed for user: {current_user.email}")
        
        return {"message": "Password changed successfully. Please log in again."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.post("/request-password-reset")
async def request_password_reset(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(get_db)
):
    """
    Request password reset (placeholder for email integration)
    """
    try:
        # Check if user exists
        stmt = select(User).where(User.email == reset_data.email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        # Always return success to prevent email enumeration
        logger.info(f"Password reset requested for: {reset_data.email}")
        
        return {
            "message": "If the email exists in our system, you will receive password reset instructions."
        }
        
    except Exception as e:
        logger.error(f"Error during password reset request: {e}")
        # Still return success to prevent information disclosure
        return {
            "message": "If the email exists in our system, you will receive password reset instructions."
        }

@router.post("/forgot-password")
async def forgot_password(
    forgot_data: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Forgot password - verify email exists
    """
    try:
        # Check if user exists
        stmt = select(User).where(User.email == forgot_data.email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Email not found in our system"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        logger.info(f"Forgot password request for: {forgot_data.email}")
        
        return {
            "message": "Email verified successfully. You can now reset your password.",
            "email": forgot_data.email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during forgot password request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forgot password request failed"
        )

@router.post("/reset-password")
async def reset_password(
    reset_data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Reset password with new password
    """
    try:
        # Check if user exists
        stmt = select(User).where(User.email == reset_data.email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Email not found in our system"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        # Set new password
        user.set_password(reset_data.new_password)
        
        # Revoke all existing refresh tokens for security
        await security.revoke_all_user_tokens(db, user.id)
        
        await db.commit()
        
        logger.info(f"Password reset for user: {user.email}")
        
        return {
            "message": "Password reset successfully. Please log in with your new password."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during password reset: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )

@router.post("/upload-avatar")
async def upload_avatar(
    avatar: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload user avatar
    """
    try:
        # Validate file type
        if not avatar.content_type or not avatar.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files are allowed"
            )
        
        # Validate file size (max 5MB)
        if avatar.size and avatar.size > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size must be less than 5MB"
            )
        
        # For now, we'll just return a placeholder URL
        # In a real implementation, you would save the file to storage
        avatar_url = f"/static/avatars/{current_user.id}.jpg"
        
        # Update user avatar URL
        current_user.avatar_url = avatar_url
        await db.commit()
        
        logger.info(f"Avatar uploaded for user: {current_user.email}")
        
        return {"avatar_url": avatar_url}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading avatar: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Avatar upload failed"
        )

@router.delete("/delete-account")
async def delete_account(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete current user account
    """
    try:
        # Revoke all tokens first
        await security.revoke_all_user_tokens(db, current_user.id)
        
        # Delete user account
        await db.delete(current_user)
        await db.commit()
        
        logger.info(f"Account deleted: {current_user.email}")
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting account: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )

# Admin routes
@router.get("/admin/users", response_model=list[UserResponse])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all users (admin only)
    """
    try:
        stmt = select(User).offset(skip).limit(min(limit, 1000))  # Cap at 1000
        result = await db.execute(stmt)
        users = result.scalars().all()
        
        return [UserResponse(**create_user_response_data(user)) for user in users]
        
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch users"
        )

@router.get("/admin/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user by ID (admin only)
    """
    try:
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(**create_user_response_data(user))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user"
        )

@router.put("/admin/users/{user_id}", response_model=UserResponse)
async def update_user_by_admin(
    user_id: str,
    user_update: AdminUserUpdate,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user by admin
    """
    try:
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Only super admin can modify other admins
        if user.is_admin and not admin_user.is_super_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super admin can modify admin users"
            )
        
        # Update user fields
        update_data = user_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"User updated by admin {admin_user.email}: {user.email}")
        return UserResponse(**create_user_response_data(user))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User update failed"
        )

@router.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    admin_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete user (super admin only)
    """
    try:
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-deletion
        if user.id == admin_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        await db.delete(user)
        await db.commit()
        
        logger.info(f"User deleted by super admin {admin_user.email}: {user.email}")
        
        return {"message": f"User {user.email} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deletion failed"
        ) 