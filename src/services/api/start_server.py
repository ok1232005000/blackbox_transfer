import uvicorn
from src.common.config import config

def main():
    host = config["server"]["host"]
    port = config["server"]["port"]
    debug = config["server"]["debug"]
    
    uvicorn.run(
        "src.services.api.main:app",
        host=host,
        port=port,
        reload=debug
    )

if __name__ == "__main__":
    main()