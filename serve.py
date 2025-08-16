import os
port = int(os.getenv("PORT", "7891"))
from app_tabs import demo
demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
