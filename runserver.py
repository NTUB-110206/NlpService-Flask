from app import app
from flask_cors import CORS

CORS(app)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
    # app.run(debug=True)