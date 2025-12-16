"""Test Ollama client response format."""
import ollama

client = ollama.Client()

print("Testing Ollama client...")
response = client.generate(
    model='sqlcoder:7b',
    prompt='Generate SQL to select top 5 products from Product table'
)

print(f"\nResponse type: {type(response)}")
print(f"\nResponse dir: {[x for x in dir(response) if not x.startswith('_')]}")

print(f"\nHas 'response' attr: {hasattr(response, 'response')}")

if hasattr(response, 'response'):
    print(f"response.response = '{response.response}'")

# Try as dict
try:
    print(f"\nresponse['response'] = '{response['response']}'")
except:
    print("\nCannot access as dict")

# Print full response
print(f"\nFull response object:\n{response}")
