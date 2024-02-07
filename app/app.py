from flask import Flask, render_template, request
from translator import Seq2SeqTransformer
from pythainlp.tokenize import Tokenizer
import torch

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mapping = torch.load('./model/vocab')['th'].get_itos()  # number-to-word mapping
inv_mapping = dict((v, k) for k, v in mapping.items())  # word-to-number mapping
tokenizer = Tokenizer(engine='newmm')

params, state = torch.load('./model/additive_Seq2SeqTransformer.pt')
model = Seq2SeqTransformer(**params, device=device).to(device)
model.load_state_dict(state)
model.eval()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        tokenized_prompt = ['<sos>'] + tokenizer.word_tokenize(prompt) + ['<eos>']  # tokenize then concatenate special tags to the start and end of list
        num_tokens = [inv_mapping(k) for k in tokenized_prompt]  # convert to numerical representations
        model_input = torch.FloatTensor(num_tokens).reshape(1, -1).to(device)  # prepare model input
        translation = model.generate(model_input)[0]

        return render_template('home.html', output=' '.join(translation), show_text="block")

    else:
        return render_template('home.html', show_text="none")

if __name__ == '__main__':
    app.run(debug=True)