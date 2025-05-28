import os
import uuid
from flask import Flask, request, render_template, make_response, redirect, url_for, session, jsonify
import identity.web
from dotenv import load_dotenv
# from agents.supervisor_langgraph import analytics_accelerator_function
from agents.superagent_finance import analytics_accelerator_function
import markdown
from utils import databases_markers, format_markdown_output

from werkzeug.middleware.proxy_fix import ProxyFix



load_dotenv()
AUTHORITY = os.getenv("AUTHORITY")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY")
CHAT_TEMPLATE = "index_v10_wcm.html"

app = Flask(__name__, template_folder='templates')
app.secret_key = SECRET_KEY
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
auth = identity.web.Auth(
    session=session,
    authority=AUTHORITY,
    client_id=CLIENT_ID,
    client_credential=CLIENT_SECRET,
)

    
# Lista para armazenar as mensagens
messages = [{'sender': 'bot', 'content': "Olá! Eu sou a LIA, sua especialista digital em insights financeiros. Como posso ajudar?"}]

@app.route("/")
def index():
    user_id = session.get('user_id')
    if user_id:
        return redirect(url_for("show_chat"))
    else:
        return redirect(url_for("login"))

@app.route('/chat', methods=['GET'])
def show_chat():
    return render_template(CHAT_TEMPLATE)

@app.route("/login")
def login():
    return render_template("login_sso_v3.html", **auth.log_in(
        scopes=["User.Read"],
        redirect_uri=url_for("auth_response", _external=True),
        prompt="select_account",
    ))

@app.route("/getAToken")
def auth_response():
    print("Começando o auth...")
    result = auth.complete_log_in(request.args)
    print(result)
    if "error" in result:
        return make_response(result.get("error"))
    
    user_id = session.get('user_id')
    if not user_id:
        session['user_id'] = str(uuid.uuid4())
        session['messages'] = {
            "ai": [],
            "user": []
        }

    print("meio do login")
    session['login'] = result.get("preferred_username")  # Email ou login do usuário
    session['name'] = result.get("name")  # Nome do usuário
    session['token'] = result.get("sub")  # Usando o "sub" como identificador único do token
    session.permanent = True  # Torna a sessão permanente (expira conforme configurado)

    return redirect(url_for("show_chat")) #, _external=True)) #, _scheme='https'))

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message')
    selected_dbs = request.json.get('databases', [])
    db_marker = databases_markers(selected_dbs)
    if user_message:
        messages.append({'sender': 'user', 'content': user_message})
        bot_response = analytics_accelerator_function(f"{db_marker} {user_message}")
        bot_response_html = markdown.markdown(bot_response)
        formatted_bot_response_html = format_markdown_output(bot_response_html)
        messages.append({'sender': 'bot', 'content': formatted_bot_response_html})
    return jsonify(messages)

@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(messages)

if __name__ == '__main__':
    app.run(debug=True)