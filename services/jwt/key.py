import json
import base64
import hashlib
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


def _int_to_base64url(n):
    """整数 → Base64URL（去掉 padding），用于 JWK 的 n / e 字段。"""
    n_bytes = n.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
    return base64.urlsafe_b64encode(n_bytes).decode('utf-8').rstrip('=')


def _rfc7638_thumbprint(jwk):
    """RFC 7638 JWK Thumbprint，用作默认 kid——同一公钥总是得到同一 kid。"""
    canonical = json.dumps(
        {"e": jwk["e"], "kty": jwk["kty"], "n": jwk["n"]},
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')
    return base64.urlsafe_b64encode(hashlib.sha256(canonical).digest()).decode('utf-8').rstrip('=')


def generate_jwt_keys(kid=None):
    """
    生成一对 JWT 用 RSA 密钥。

    参数:
      kid: 可选，自定义 key id。不传则使用 RFC 7638 thumbprint 作为 kid。

    返回: (kid, 私钥 PEM 字符串, JWK 公钥字典——单条，未装进 keys 数组)
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_numbers = private_key.public_key().public_numbers()

    jwk = {
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "n": _int_to_base64url(public_numbers.n),
        "e": _int_to_base64url(public_numbers.e),
    }
    jwk["kid"] = kid or _rfc7638_thumbprint(jwk)

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return jwk["kid"], private_pem.decode('utf-8'), jwk


# ---------- JWKS 文件读写（多 kid） ----------

def load_jwks(jwks_file='jwks.json'):
    """读取已有 JWKS；文件不存在时返回空集合 {'keys': []}。"""
    path = Path(jwks_file)
    if not path.exists():
        return {"keys": []}
    with path.open('r') as f:
        data = json.load(f)
    # 兼容历史数据：单 key 但缺 kid 时，补上 thumbprint
    for k in data.get('keys', []):
        if 'kid' not in k and k.get('kty') == 'RSA':
            k['kid'] = _rfc7638_thumbprint(k)
    return data


def save_jwks(jwks, jwks_file='jwks.json'):
    """覆盖写入 JWKS。"""
    with open(jwks_file, 'w') as f:
        json.dump(jwks, f, indent=2)
    print(f"✅ JWKS 已保存到: {jwks_file}（共 {len(jwks['keys'])} 个 key）")


def add_key_to_jwks(jwk, jwks_file='jwks.json'):
    """
    把一条 JWK 追加进 JWKS 文件，按 kid 去重（同 kid 会被新条目覆盖）。
    """
    jwks = load_jwks(jwks_file)
    jwks['keys'] = [k for k in jwks['keys'] if k.get('kid') != jwk.get('kid')]
    jwks['keys'].append(jwk)
    save_jwks(jwks, jwks_file)
    return jwks


def remove_key_from_jwks(kid, jwks_file='jwks.json'):
    """按 kid 从 JWKS 中删一条（key 轮换下线场景）。"""
    jwks = load_jwks(jwks_file)
    before = len(jwks['keys'])
    jwks['keys'] = [k for k in jwks['keys'] if k.get('kid') != kid]
    if len(jwks['keys']) == before:
        print(f"⚠️  未找到 kid={kid}，跳过")
    else:
        save_jwks(jwks, jwks_file)
    return jwks


def list_kids(jwks_file='jwks.json'):
    """列出 JWKS 中所有 kid。"""
    return [k.get('kid') for k in load_jwks(jwks_file).get('keys', [])]


# ---------- 私钥读写（按 kid） ----------

def save_private_key(private_pem, kid, private_dir='private_keys'):
    """私钥落盘到 <private_dir>/<kid>.key。"""
    dir_path = Path(private_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{kid}.key"
    with path.open('w') as f:
        f.write(private_pem)
    print(f"✅ 私钥已保存到: {path}")
    return path


def load_private_key_by_kid(kid, private_dir='private_keys'):
    """按 kid 加载私钥（用于签发）。"""
    path = Path(private_dir) / f"{kid}.key"
    with path.open('r') as f:
        private_pem = f.read()
    return serialization.load_pem_private_key(
        private_pem.encode('utf-8'),
        password=None,
        backend=default_backend(),
    )


def load_private_key_from_file(private_file='private.key'):
    """旧入口：从给定路径加载私钥（保留以兼容历史调用）。"""
    with open(private_file, 'r') as f:
        private_pem = f.read()
    return serialization.load_pem_private_key(
        private_pem.encode('utf-8'),
        password=None,
        backend=default_backend(),
    )


# ---------- 一站式入口 ----------

def add_new_key(jwks_file='jwks.json', private_dir='private_keys', kid=None):
    """
    生成 → 追加进 JWKS → 私钥落盘，一步到位。
    返回 (kid, jwk)。
    """
    kid, private_pem, jwk = generate_jwt_keys(kid=kid)
    add_key_to_jwks(jwk, jwks_file=jwks_file)
    save_private_key(private_pem, kid, private_dir=private_dir)
    print(f"✅ 新增 JWK kid={kid}")
    return kid, jwk


# ---------- 签发 / 验证 ----------

def create_jwt_token(user_id, username, private_key, kid=None, expires_in_hours=1):
    """
    签发 JWT；传入 kid 时写入 header，验证端按 kid 选择公钥。
    """
    import jwt
    from datetime import datetime, timedelta

    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'iat': datetime.utcnow(),
    }
    headers = {'kid': kid} if kid else None
    return jwt.encode(payload, private_key, algorithm='RS256', headers=headers)


def verify_jwt_token(token, jwks_dict):
    """
    验证 JWT。PyJWKSet 原生支持多 kid——按 JWT header 里的 kid
    从 jwks_dict['keys'] 中挑对应公钥；header 没带 kid 时回退到唯一 key。
    """
    import jwt
    from jwt import PyJWKSet

    try:
        jwks_set = PyJWKSet.from_dict(jwks_dict)
        header = jwt.get_unverified_header(token)
        kid = header.get('kid')
        if kid is not None:
            signing_key = jwks_set[kid].key
        elif len(jwks_set.keys) == 1:
            signing_key = jwks_set.keys[0].key
        else:
            raise jwt.InvalidTokenError(
                "JWT header 未携带 kid，且 JWKS 中存在多个 key，无法选择"
            )
        return jwt.decode(
            token,
            signing_key,
            algorithms=['RS256'],
            options={'require': ['exp', 'iat']},
        )
    except KeyError as exc:
        print(f"Token 无效: JWKS 中找不到 kid={exc}")
        return None
    except jwt.ExpiredSignatureError:
        print("Token 已过期")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Token 无效: {e}")
        return None


# ---------- 历史接口（保留，避免破坏旧调用方） ----------

def save_keys(private_pem, jwks, private_file='private.key', jwks_file='jwks.json'):
    """旧接口：单 key 直接落盘到固定文件名。新代码请用 add_new_key()。"""
    with open(private_file, 'w') as f:
        f.write(private_pem)
    print(f"✅ 私钥已保存到: {private_file}")
    save_jwks(jwks, jwks_file)


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("🔑 演示多 kid JWKS：生成两组密钥对\n")

    kid_a, _ = add_new_key()
    kid_b, _ = add_new_key()

    print("\n" + "=" * 50)
    print("当前 JWKS（粘进阿里云函数计算公钥配置即可）:")
    print("=" * 50)
    print(json.dumps(load_jwks(), indent=2))
    print(f"\nkids: {list_kids()}")

    # —— 用 kid_a 签 token，再让验证端根据 header.kid 自动找回 ——
    print("\n" + "=" * 50)
    print(f"用 kid={kid_a} 签发 → 验证")
    print("=" * 50)
    token_a = create_jwt_token(
        user_id=123,
        username="alice",
        private_key=load_private_key_by_kid(kid_a),
        kid=kid_a,
        expires_in_hours=36,
    )
    print(f"📝 Token:\n{token_a}")
    payload_a = verify_jwt_token(token_a, load_jwks())
    print(f"✅ 验证: {payload_a}")

    # —— 用 kid_b 签同样能验证，证明多 kid 各自独立工作 ——
    print("\n" + "=" * 50)
    print(f"用 kid={kid_b} 签发 → 验证")
    print("=" * 50)
    token_b = create_jwt_token(
        user_id=456,
        username="bob",
        private_key=load_private_key_by_kid(kid_b),
        kid=kid_b,
        expires_in_hours=2,
    )
    print(f"📝 Token:\n{token_b}")
    payload_b = verify_jwt_token(token_b, load_jwks())
    print(f"✅ 验证: {payload_b}")
