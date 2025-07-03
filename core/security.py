from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import secrets
import uuid
import logging

from .config import settings
from .database import get_db
from models.user import User, RefreshToken, UserRole
from models.admin import Admin, AdminRefreshToken, AdminRole

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass

class SecurityManager:
    """Centralized security manager for authentication and authorization"""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = settings.refresh_token_expire_days
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise AuthenticationError("Failed to create access token")
    
    def create_refresh_token(self) -> str:
        """Create secure refresh token"""
        return secrets.token_urlsafe(32)
    
    async def create_refresh_token_record(
        self,
        db: AsyncSession,
        user_id: Union[str, uuid.UUID],
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> RefreshToken:
        """Create refresh token database record"""
        token = self.create_refresh_token()
        expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        refresh_token = RefreshToken(
            token=token,
            user_id=user_id,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        db.add(refresh_token)
        await db.commit()
        await db.refresh(refresh_token)
        
        return refresh_token
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                raise AuthenticationError(f"Invalid token type. Expected {token_type}")
            
            # Check expiration
            exp = payload.get("exp")
            if exp is None:
                raise AuthenticationError("Token missing expiration")
            
            if datetime.utcnow() > datetime.fromtimestamp(exp):
                raise AuthenticationError("Token has expired")
            
            return payload
            
        except JWTError as e:
            logger.error(f"JWT verification error: {e}")
            raise AuthenticationError("Invalid token")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise AuthenticationError("Token verification failed")
    
    async def verify_refresh_token(self, db: AsyncSession, token: str) -> RefreshToken:
        """Verify refresh token from database"""
        try:
            stmt = select(RefreshToken).where(
                RefreshToken.token == token,
                RefreshToken.is_revoked == False
            )
            result = await db.execute(stmt)
            refresh_token = result.scalar_one_or_none()
            
            if not refresh_token:
                raise AuthenticationError("Invalid refresh token")
            
            if refresh_token.is_expired:
                raise AuthenticationError("Refresh token has expired")
            
            # Update last used timestamp
            refresh_token.update_last_used()
            await db.commit()
            
            return refresh_token
            
        except Exception as e:
            logger.error(f"Refresh token verification error: {e}")
            raise AuthenticationError("Refresh token verification failed")
    
    async def revoke_refresh_token(self, db: AsyncSession, token: str) -> bool:
        """Revoke refresh token"""
        try:
            stmt = select(RefreshToken).where(RefreshToken.token == token)
            result = await db.execute(stmt)
            refresh_token = result.scalar_one_or_none()
            
            if refresh_token:
                refresh_token.revoke()
                await db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking refresh token: {e}")
            return False
    
    async def revoke_all_user_tokens(self, db: AsyncSession, user_id: Union[str, uuid.UUID]) -> int:
        """Revoke all refresh tokens for a user"""
        try:
            stmt = select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False
            )
            result = await db.execute(stmt)
            tokens = result.scalars().all()
            
            count = 0
            for token in tokens:
                token.revoke()
                count += 1
            
            await db.commit()
            return count
            
        except Exception as e:
            logger.error(f"Error revoking user tokens: {e}")
            return 0
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        try:
            stmt = select(User).where(User.email == email)
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            # Check if account is locked
            if user.is_locked:
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail=f"Account is locked until {user.locked_until}"
                )
            
            # Verify password
            if not user.verify_password(password):
                # Increment failed login attempts
                user.increment_failed_login()
                await db.commit()
                return None
            
            # Check if user is active and verified
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated"
                )
            
            # Reset failed login attempts on successful login
            user.reset_failed_login()
            await db.commit()
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def get_current_user(self, db: AsyncSession, token: str) -> User:
        """Get current user from JWT token"""
        logger.info(f"SecurityManager.get_current_user called with token length: {len(token)}")
        
        try:
            logger.info("Verifying JWT token...")
            payload = self.verify_token(token)
            logger.info(f"Token payload: {payload}")
            
            user_id = payload.get("sub")
            logger.info(f"Extracted user_id: {user_id}")
            
            if user_id is None:
                logger.error("Token missing user ID in 'sub' field")
                raise AuthenticationError("Token missing user ID")
            
            logger.info(f"Querying database for user: {user_id}")
            stmt = select(User).where(User.id == user_id)
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            
            if user is None:
                logger.error(f"User not found in database: {user_id}")
                raise AuthenticationError("User not found")
            
            logger.info(f"User found: {user.email}, active: {user.is_active}")
            
            if not user.is_active:
                logger.error(f"User account is deactivated: {user.email}")
                raise AuthenticationError("User account is deactivated")
            
            logger.info(f"Authentication successful for user: {user.email}")
            return user
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting current user: {type(e).__name__}: {str(e)}")
            logger.exception("Full traceback:")
            raise AuthenticationError("Failed to get current user")
    
    def check_permissions(self, user: User, required_role: UserRole) -> bool:
        """Check if user has required role permissions"""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.MODERATOR: 2,
            UserRole.ADMIN: 3,
            UserRole.SUPER_ADMIN: 4
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    def require_permissions(self, user: User, required_role: UserRole) -> None:
        """Require user to have specific role permissions"""
        if not self.check_permissions(user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}"
            )

    async def authenticate_admin(self, db: AsyncSession, email: str, password: str) -> Optional[Admin]:
        """Authenticate admin with email and password"""
        try:
            stmt = select(Admin).where(Admin.email == email)
            result = await db.execute(stmt)
            admin = result.scalar_one_or_none()
            
            if not admin:
                return None
            
            # Check if account is locked
            if admin.is_locked:
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail=f"Account is locked until {admin.locked_until}"
                )
            
            # Verify password
            if not admin.verify_password(password):
                # Increment failed login attempts
                admin.increment_failed_login()
                await db.commit()
                return None
            
            # Check if admin is active and verified
            if not admin.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin account is deactivated"
                )
            
            # Reset failed login attempts on successful login
            admin.reset_failed_login()
            await db.commit()
            
            return admin
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Admin authentication error: {e}")
            return None

    async def get_current_admin(self, db: AsyncSession, token: str) -> Admin:
        """Get current admin from JWT token"""
        try:
            payload = self.verify_token(token)
            admin_id = payload.get("sub")
            token_type = payload.get("type")
            
            if admin_id is None:
                raise AuthenticationError("Token missing admin ID")
            
            # Check if token is specifically for admin
            if token_type != "admin":
                raise AuthenticationError("Invalid token type for admin")
            
            stmt = select(Admin).where(Admin.id == admin_id)
            result = await db.execute(stmt)
            admin = result.scalar_one_or_none()
            
            if admin is None:
                raise AuthenticationError("Admin not found")
            
            if not admin.is_active:
                raise AuthenticationError("Admin account is deactivated")
            
            return admin
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Error getting current admin: {e}")
            raise AuthenticationError("Failed to get current admin")

    async def create_admin_refresh_token_record(
        self,
        db: AsyncSession,
        admin_id: Union[str, uuid.UUID],
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> AdminRefreshToken:
        """Create admin refresh token database record"""
        token = self.create_refresh_token()
        expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        refresh_token = AdminRefreshToken(
            token=token,
            admin_id=admin_id,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        db.add(refresh_token)
        await db.commit()
        await db.refresh(refresh_token)
        
        return refresh_token

    async def verify_admin_refresh_token(self, db: AsyncSession, token: str) -> AdminRefreshToken:
        """Verify admin refresh token from database"""
        try:
            stmt = select(AdminRefreshToken).where(
                AdminRefreshToken.token == token,
                AdminRefreshToken.is_revoked == False
            )
            result = await db.execute(stmt)
            refresh_token = result.scalar_one_or_none()
            
            if not refresh_token:
                raise AuthenticationError("Invalid admin refresh token")
            
            if refresh_token.is_expired:
                raise AuthenticationError("Admin refresh token has expired")
            
            # Update last used timestamp
            refresh_token.update_last_used()
            await db.commit()
            
            return refresh_token
            
        except Exception as e:
            logger.error(f"Admin refresh token verification error: {e}")
            raise AuthenticationError("Admin refresh token verification failed")

    async def revoke_admin_refresh_token(self, db: AsyncSession, token: str) -> bool:
        """Revoke admin refresh token"""
        try:
            stmt = select(AdminRefreshToken).where(AdminRefreshToken.token == token)
            result = await db.execute(stmt)
            refresh_token = result.scalar_one_or_none()
            
            if refresh_token:
                refresh_token.revoke()
                await db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking admin refresh token: {e}")
            return False

    async def revoke_all_admin_tokens(self, db: AsyncSession, admin_id: Union[str, uuid.UUID]) -> int:
        """Revoke all refresh tokens for an admin"""
        try:
            stmt = select(AdminRefreshToken).where(
                AdminRefreshToken.admin_id == admin_id,
                AdminRefreshToken.is_revoked == False
            )
            result = await db.execute(stmt)
            tokens = result.scalars().all()
            
            count = 0
            for token in tokens:
                token.revoke()
                count += 1
            
            await db.commit()
            return count
            
        except Exception as e:
            logger.error(f"Error revoking admin tokens: {e}")
            return 0

# Global security manager instance
security = SecurityManager()

# HTTP Bearer token scheme
security_scheme = HTTPBearer(auto_error=False)  # Don't auto-error for optional auth

# FastAPI Dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """FastAPI dependency to get current authenticated user"""
    logger.info("Authentication attempt:")
    logger.info(f"  - Credentials provided: {credentials is not None}")
    
    if credentials:
        token_preview = credentials.credentials[:20] + "..." if len(credentials.credentials) > 20 else credentials.credentials
        logger.info(f"  - Token preview: {token_preview}")
        logger.info(f"  - Token length: {len(credentials.credentials)}")
    
    try:
        token = credentials.credentials
        logger.info("Calling security.get_current_user...")
        user = await security.get_current_user(db, token)
        logger.info(f"Authentication successful for user: {user.email} (ID: {user.id})")
        return user
    except AuthenticationError as e:
        logger.error(f"AuthenticationError: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Unexpected authentication error: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> Admin:
    """FastAPI dependency to get current authenticated admin"""
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
    except Exception as e:
        logger.error(f"Admin authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate admin credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """FastAPI dependency to get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user = await security.get_current_user(db, token)
        return user
    except:
        return None

async def get_optional_user_with_query_token(
    token: Optional[str] = Query(None, description="Authentication token for direct access"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """FastAPI dependency to get current user from Bearer header OR query parameter"""
    print(f"🔍 AUTH CHECK: header_creds={credentials is not None}, query_token={token is not None}")
    
    # Try Bearer token first
    if credentials:
        try:
            print("Trying Bearer token authentication...")
            user = await security.get_current_user(db, credentials.credentials)
            print(f"Bearer auth successful: {user.email}")
            return user
        except Exception as e:
            print(f"Bearer auth failed: {e}")
    
    # Try query parameter token
    if token:
        try:
            print("Trying query parameter token authentication...")
            user = await security.get_current_user(db, token)
            print(f"Query token auth successful: {user.email}")
            return user
        except Exception as e:
            print(f"Query token auth failed: {e}")
    
    print("No valid authentication found")
    return None

# Role-based permission decorators
def require_role(required_role: UserRole):
    """Decorator to require specific user role"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        security.require_permissions(current_user, required_role)
        return current_user
    return role_checker

def require_admin_role(required_role: AdminRole):
    """Decorator to require specific admin role"""
    async def admin_role_checker(current_admin: Admin = Depends(get_current_admin)) -> Admin:
        if not current_admin.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient admin privileges"
            )
        return current_admin
    return admin_role_checker 