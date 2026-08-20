import jwt
import logging
from typing import Optional, List
# pyrefly: ignore [missing-import]
from src.auth.models import AuthContext
from src.config import load_config

logger = logging.getLogger(__name__)

class JWTValidator:
    def __init__(self):
        # We load config dynamically on validate to avoid global state issues 
        pass

    def validate(self, token: str) -> Optional[AuthContext]:
        cfg = load_config()
        jwt_cfg = cfg.auth.jwt
        
        if not jwt_cfg or not jwt_cfg.enabled:
            # Fallback when JWT is disabled - just return None
            return None
            
        try:
            # jwt.decode checks signature, exp, iss, and aud automatically if provided
            options = {
                "verify_signature": True,
                "verify_exp": True,
                "verify_iss": bool(jwt_cfg.issuer),
                "verify_aud": bool(jwt_cfg.audience),
            }
            
            if jwt_cfg.jwks_url:
                from jwt import PyJWKClient
                jwks_client = PyJWKClient(jwt_cfg.jwks_url)
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                key = signing_key.key
            else:
                key = jwt_cfg.secret_key
                if not key:
                    logger.error("JWT secret_key or jwks_url is not configured.")
                    return None

            claims = jwt.decode(
                token,
                key=key,
                algorithms=jwt_cfg.algorithms,
                issuer=jwt_cfg.issuer if jwt_cfg.issuer else None,
                audience=jwt_cfg.audience if jwt_cfg.audience else None,
                options=options
            )
            
            sub = claims.get("sub")
            if not sub:
                logger.error("JWT missing 'sub' claim.")
                return None
                
            return AuthContext(
                subject=str(sub),
                issuer=claims.get("iss", ""),
                audience=claims.get("aud", ""),
                claims=claims,
                token=token
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT validation failed: Token has expired.")
            return None
        except jwt.InvalidIssuerError:
            logger.warning("JWT validation failed: Invalid issuer.")
            return None
        except jwt.InvalidAudienceError:
            logger.warning("JWT validation failed: Invalid audience.")
            return None
        except jwt.InvalidSignatureError:
            logger.warning("JWT validation failed: Invalid signature.")
            return None
        except jwt.DecodeError as e:
            logger.warning(f"JWT validation failed: Decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT validation error: {e}")
            return None

default_jwt_validator = JWTValidator()
