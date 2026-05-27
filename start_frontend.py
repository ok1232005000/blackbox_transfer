import subprocess
import sys

def main():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/services/frontend/app.py",
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()