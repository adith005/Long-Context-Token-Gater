import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# List all keys
keys = r.keys('*')
print("Keys:", keys)

# View data for each key
for key in keys:
    key_type = r.type(key).decode()
    print(f"\nKey: {key.decode()} | Type: {key_type}")
    if key_type == "string":
        val = r.get(key).decode()
        print("Value (JSON):", val)