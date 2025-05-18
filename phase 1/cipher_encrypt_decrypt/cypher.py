
#libraries of the RSA 
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes

#libraries of the AES
from Crypto.Cipher import AES


# RSA encryption and decryption
    
def generate_key():
    key = RSA.generate(2048)
    private_key = key.export_key(passphrase="password")
    file_out = open("private.pem", "wb")
    file_out.write(private_key)
    file_out.close()

    public_key = key.publickey().export_key()
    file_out = open("receiver.pem", "wb")
    file_out.write(public_key)
    file_out.close()
    
def encrypt(message):
    file_in = open("receiver.pem", "rb")
    public_key = RSA.import_key(file_in.read())
    cipher = PKCS1_OAEP.new(public_key)
    encrypted = cipher.encrypt(message)
    return encrypted

def decrypt(encrypted, password="password"):
    file_in = open("private.pem", "rb")
    private_key = RSA.import_key(file_in.read(), passphrase=password)
    cipher = PKCS1_OAEP.new(private_key)
    decrypted = cipher.decrypt(encrypted)
    return decrypted

#AES encryption and decryption

# Encrypt data
def aes_encrypt(data, key):
    cipher = AES.new(key, AES.MODE_GCM)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag

# Decrypt data
def aes_decrypt(nonce, ciphertext, tag, key):
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return data



    
    
        