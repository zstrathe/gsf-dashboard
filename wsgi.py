from src.dash_app.app import app
application = app.server

if __name__ == '__main__':
    application.run()