"""Security utilities for OpenSynthetics."""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from loguru import logger


@dataclass
class APIKey:
    """API key data structure."""
    id: str
    name: str
    service: str
    key_value: str
    status: str = "active"
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    rate_limit: Optional[int] = None
    scope: List[str] = None
    description: Optional[str] = None
    usage_24h: int = 0
    usage_month: int = 0
    failed_requests_24h: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.scope is None:
            self.scope = ["read"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIKey':
        """Create from dictionary with proper datetime deserialization."""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'expires_at', 'last_used']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if the API key is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_rate_limited(self) -> bool:
        """Check if the API key has exceeded its rate limit."""
        if not self.rate_limit:
            return False
        return self.usage_24h >= (self.rate_limit * 24)  # Convert hourly to daily
    
    def increment_usage(self):
        """Increment usage counters."""
        self.usage_24h += 1
        self.usage_month += 1
        self.last_used = datetime.utcnow()
    
    def increment_failed_requests(self):
        """Increment failed request counter."""
        self.failed_requests_24h += 1


class SecurityManager:
    """Manages security operations for OpenSynthetics."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize security manager."""
        self.config_dir = config_dir or Path.home() / ".opensynthetics"
        self.config_dir.mkdir(exist_ok=True)
        
        self.api_keys_file = self.config_dir / "api_keys.json"
        self.credentials_file = self.config_dir / "credentials.json"
        self.security_config_file = self.config_dir / "security.json"
        
        self._api_keys: Dict[str, APIKey] = {}
        self._credentials: Dict[str, Dict[str, Any]] = {}
        self._security_config: Dict[str, Any] = {}
        
        self.load_all()
    
    def load_all(self):
        """Load all security data from files."""
        self.load_api_keys()
        self.load_credentials()
        self.load_security_config()
    
    def load_api_keys(self):
        """Load API keys from file."""
        if self.api_keys_file.exists():
            try:
                with open(self.api_keys_file, 'r') as f:
                    data = json.load(f)
                    self._api_keys = {
                        key_id: APIKey.from_dict(key_data)
                        for key_id, key_data in data.items()
                    }
                logger.info(f"Loaded {len(self._api_keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
                self._api_keys = {}
    
    def save_api_keys(self):
        """Save API keys to file."""
        try:
            data = {
                key_id: key.to_dict()
                for key_id, key in self._api_keys.items()
            }
            with open(self.api_keys_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("API keys saved successfully")
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
    
    def load_credentials(self):
        """Load credentials from file."""
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    self._credentials = json.load(f)
                logger.info(f"Loaded credentials for {len(self._credentials)} services")
            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                self._credentials = {}
    
    def save_credentials(self):
        """Save credentials to file."""
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(self._credentials, f, indent=2)
            logger.debug("Credentials saved successfully")
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def load_security_config(self):
        """Load security configuration."""
        if self.security_config_file.exists():
            try:
                with open(self.security_config_file, 'r') as f:
                    self._security_config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load security config: {e}")
                self._security_config = {}
        
        # Set defaults
        if not self._security_config:
            self._security_config = {
                "default_rate_limit": 1000,
                "default_expiration_days": 365,
                "require_api_key": True,
                "allowed_origins": ["*"],
                "session_timeout": 3600
            }
            self.save_security_config()
    
    def save_security_config(self):
        """Save security configuration."""
        try:
            with open(self.security_config_file, 'w') as f:
                json.dump(self._security_config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save security config: {e}")
    
    def generate_api_key(self, prefix: str = "os") -> str:
        """Generate a new API key."""
        return f"{prefix}_{secrets.token_urlsafe(32)}"
    
    def create_api_key(
        self,
        name: str,
        service: str,
        key_value: Optional[str] = None,
        expiration_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        scope: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> APIKey:
        """Create a new API key."""
        key_id = secrets.token_urlsafe(16)
        
        if not key_value:
            key_value = self.generate_api_key()
        
        expires_at = None
        if expiration_days:
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)
        
        api_key = APIKey(
            id=key_id,
            name=name,
            service=service,
            key_value=key_value,
            expires_at=expires_at,
            rate_limit=rate_limit or self._security_config.get("default_rate_limit"),
            scope=scope or ["read"],
            description=description
        )
        
        self._api_keys[key_id] = api_key
        self.save_api_keys()
        
        logger.info(f"Created API key '{name}' for service '{service}'")
        return api_key
    
    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._api_keys.get(key_id)
    
    def get_api_key_by_value(self, key_value: str) -> Optional[APIKey]:
        """Get API key by value."""
        for api_key in self._api_keys.values():
            if api_key.key_value == key_value:
                return api_key
        return None
    
    def list_api_keys(self) -> List[APIKey]:
        """List all API keys."""
        return list(self._api_keys.values())
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._api_keys:
            self._api_keys[key_id].status = "revoked"
            self.save_api_keys()
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False
    
    def verify_api_key(self, key_value: str) -> Optional[APIKey]:
        """Verify an API key and return it if valid."""
        api_key = self.get_api_key_by_value(key_value)
        
        if not api_key:
            return None
        
        if api_key.status != "active":
            return None
        
        if api_key.is_expired():
            api_key.status = "expired"
            self.save_api_keys()
            return None
        
        if api_key.is_rate_limited():
            logger.warning(f"API key {api_key.id} rate limit exceeded")
            api_key.increment_failed_requests()
            self.save_api_keys()
            return None
        
        api_key.increment_usage()
        self.save_api_keys()
        
        return api_key
    
    def store_credentials(self, service: str, credentials: Dict[str, Any]):
        """Store credentials for a service."""
        # Encrypt sensitive data in production
        self._credentials[service] = credentials
        self.save_credentials()
        logger.info(f"Stored credentials for service: {service}")
    
    def get_credentials(self, service: str) -> Optional[Dict[str, Any]]:
        """Get credentials for a service."""
        return self._credentials.get(service)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a password with salt."""
        if not salt:
            salt = secrets.token_hex(16)
        
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return hashed.hex(), salt
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify a password against its hash."""
        test_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(test_hash, hashed)
    
    def generate_session_token(self) -> str:
        """Generate a session token."""
        return secrets.token_urlsafe(32)
    
    def cleanup_expired_keys(self):
        """Clean up expired API keys."""
        expired_keys = [
            key_id for key_id, key in self._api_keys.items()
            if key.is_expired()
        ]
        
        for key_id in expired_keys:
            self._api_keys[key_id].status = "expired"
        
        if expired_keys:
            self.save_api_keys()
            logger.info(f"Marked {len(expired_keys)} API keys as expired")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all API keys."""
        total_requests_today = sum(key.usage_24h for key in self._api_keys.values())
        total_requests_month = sum(key.usage_month for key in self._api_keys.values())
        active_keys = sum(1 for key in self._api_keys.values() if key.status == "active")
        failed_requests = sum(key.failed_requests_24h for key in self._api_keys.values())
        
        return {
            "requests_today": total_requests_today,
            "requests_month": total_requests_month,
            "active_keys": active_keys,
            "failed_requests": failed_requests,
            "total_keys": len(self._api_keys)
        }
    
    def mask_api_key(self, key_value: str) -> str:
        """Mask an API key for display purposes."""
        if len(key_value) <= 8:
            return "*" * len(key_value)
        return key_value[:4] + "*" * (len(key_value) - 8) + key_value[-4:]


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def verify_api_key(key_value: str) -> Optional[APIKey]:
    """Verify an API key using the global security manager."""
    return get_security_manager().verify_api_key(key_value)


def get_current_user(api_key: str) -> Optional[Dict[str, Any]]:
    """Get current user information from API key."""
    security_manager = get_security_manager()
    key = security_manager.verify_api_key(api_key)
    
    if key:
        return {
            "id": key.id,
            "name": key.name,
            "service": key.service,
            "scope": key.scope
        }
    
    return None


def create_default_api_key() -> str:
    """Create a default API key for initial setup."""
    security_manager = get_security_manager()
    
    # Check if any active keys exist
    active_keys = [key for key in security_manager.list_api_keys() if key.status == "active"]
    
    if not active_keys:
        api_key = security_manager.create_api_key(
            name="Default API Key",
            service="opensynthetics",
            scope=["read", "write", "admin"],
            description="Default API key created during initial setup"
        )
        logger.info(f"Created default API key: {security_manager.mask_api_key(api_key.key_value)}")
        return api_key.key_value
    
    return active_keys[0].key_value 