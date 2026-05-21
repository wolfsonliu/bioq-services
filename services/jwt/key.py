import json
import base64
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def generate_jwt_keys():
    """
    生成 JWT 所需的 RSA 密钥对
    返回: (私钥 PEM 字符串, JWKS 格式的公钥字典)
    """
    
    # 1. 生成 RSA 密钥对（2048位）
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    
    # 2. 导出私钥为 PEM 格式（PKCS#8 标准）
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()  # 不加密，如需密码保护可改为 BestAvailableEncryption(b'password')
    )
    
    # 3. 导出公钥为 JWKS 格式
    # 获取公钥的 RSA 参数
    public_numbers = public_key.public_numbers()
    
    # 将整数转换为 Base64URL 编码
    def int_to_base64url(n):
        # 转换为字节（大端序）
        n_bytes = n.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
        # Base64 编码并转换为 URL-safe 格式
        return base64.urlsafe_b64encode(n_bytes).decode('utf-8').rstrip('=')
    
    jwks = {
        "keys": [
            {
                "kty": "RSA",           # 密钥类型
                "alg": "RS256",         # 算法
                "use": "sig",           # 用途：签名
                "n": int_to_base64url(public_numbers.n),  # modulus
                "e": int_to_base64url(public_numbers.e),  # exponent
            }
        ]
    }
    
    return private_pem.decode('utf-8'), jwks

def save_keys(private_pem, jwks, private_file='private.key', jwks_file='jwks.json'):
    """
    保存密钥到文件
    """
    # 保存私钥
    with open(private_file, 'w') as f:
        f.write(private_pem)
    print(f"✅ 私钥已保存到: {private_file}")
    
    # 保存 JWKS 公钥配置
    with open(jwks_file, 'w') as f:
        json.dump(jwks, f, indent=2)
    print(f"✅ JWKS 公钥配置已保存到: {jwks_file}")
    
    # 打印预览
    print("\n" + "="*50)
    print("私钥内容（保存到 private.key）:")
    print("="*50)
    print(private_pem)
    
    print("\n" + "="*50)
    print("JWKS 公钥配置（复制到阿里云函数计算）:")
    print("="*50)
    print(json.dumps(jwks, indent=2))

def load_private_key_from_file(private_file='private.key'):
    """
    从文件加载私钥（用于签发 JWT）
    """
    with open(private_file, 'r') as f:
        private_pem = f.read()
    
    return serialization.load_pem_private_key(
        private_pem.encode('utf-8'),
        password=None,  # 如果私钥有密码保护，这里传入密码字节串
        backend=default_backend()
    )

def create_jwt_token(user_id, username, private_key, expires_in_hours=1):
    """
    使用生成的私钥签发 JWT Token
    需要安装: pip install pyjwt
    """
    import jwt
    from datetime import datetime, timedelta
    
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'iat': datetime.utcnow()  # 签发时间
    }
    
    # 使用 RS256 算法签名
    token = jwt.encode(
        payload,
        private_key,
        algorithm='RS256'
    )
    
    return token

def verify_jwt_token(token, jwks_dict):
    """
    验证 JWT Token（使用 JWKS 公钥）
    需要安装: pip install pyjwt
    """
    import jwt
    from jwt import PyJWKClient
    
    # 从 JWKS 创建验证客户端
    jwks_client = PyJWKClient.from_dict(jwks_dict)
    
    try:
        # 获取签名密钥
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        # 验证 token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=['RS256'],
            options={'require': ['exp', 'iat']}
        )
        return payload
    except jwt.ExpiredSignatureError:
        print("Token 已过期")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Token 无效: {e}")
        return None

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 生成密钥对
    print("🔑 正在生成 RSA 密钥对...\n")
    private_pem, jwks = generate_jwt_keys()
    
    # 2. 保存到文件
    save_keys(private_pem, jwks)
    
    # 3. 演示签发和验证 JWT
    print("\n" + "="*50)
    print("测试 JWT 签发和验证")
    print("="*50)
    
    # 加载私钥
    private_key = load_private_key_from_file('private.key')
    
    # 签发 token
    token = create_jwt_token(
        user_id=123,
        username="test_user",
        private_key=private_key,
        expires_in_hours=2
    )
    print(f"\n📝 生成的 JWT Token:\n{token}")
    
    # 验证 token
    print("\n✅ 验证 Token...")
    payload = verify_jwt_token(token, jwks)
    if payload:
        print(f"验证成功！解析内容: {payload}")
