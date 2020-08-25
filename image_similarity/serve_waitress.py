# Serve the flask app with wsgi using waitress

from waitress import serve
import web_app

serve(web_app.app, port=9001, threads=6)
