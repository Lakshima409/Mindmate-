import torch
print('torch version:', torch.__version__)

data = torch.load("data/data.pth", map_location=torch.device("cpu"))
print('Keys:', list(data.keys()))
print('Tags:', data.get('tags', 'MISSING'))
print('Input size:', data.get('input_size', 'MISSING'))
print('Hidden size:', data.get('hidden_size', 'MISSING'))
print('Output size:', data.get('output_size', 'MISSING'))
print('All words count:', len(data.get('all_words', [])))
print('First 10 words:', data.get('all_words', [])[:10] if data.get('all_words') else 'MISSING')
print()
print('Model loaded successfully!')