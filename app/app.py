from flask import Flask, render_template, request
from translator import *
from torchtext.data.utils import get_tokenizer
import torch

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_transform = torch.load('./models/vocab')['en']  # For tokenizing english text to numerical tokens
mapping = torch.load('./models/vocab')['th'].get_itos()  # For transforming numerical output to Thai text
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# params, state = torch.load('./models/additive_Seq2SeqTransformer.pt')
# model = Seq2SeqTransformer(**params, device=device).to(device)
# model.load_state_dict(state)
# model.eval()

checkpoint = torch.load('./models/additive_Seq2SeqTransformer.pt')
params = checkpoint['hyperparameters']

enc = Encoder(**params, device=device)
dec = Decoder(**params, device=device)

enc.load_state_dict(checkpoint['encoder'])
dec.load_state_dict(checkpoint['decoder'])

model = Seq2SeqTransformer(**params, encoder=enc, decoder=dec, device=device).to(device)
model.load_state_dict(checkpoint['model'])

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt'].lower()
        tokenized_prompt = ['<sos>'] + tokenizer(prompt) + ['<eos>']  # tokenize then concatenate special tags to the start and end of list
        num_tokens = vocab_transform(tokenized_prompt)  # convert to numerical representations
        print(num_tokens)
        model_input = torch.tensor(num_tokens, dtype=torch.int64).reshape(1, -1).to(device)  # prepare model input
        model_output = model.generate(model_input)[0]
        print(model_output)
        translation = [mapping[token.item()] for token in model_output]

        return render_template('home.html', output=' '.join(translation), show_text="block")

    else:
        return render_template('home.html', show_text="none")

if __name__ == '__main__':
    app.run(debug=True)