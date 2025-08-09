from flask import Flask, render_template, request
from src.predictor import get_signal

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    signal = None
    ticker = None
    if request.method == 'POST':
        ticker = request.form['ticker']
        if ticker:
            signal = get_signal(ticker.upper())
    return render_template('index.html', signal=signal, ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
