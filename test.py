import requests

def get_output_dir_from_api(host="127.0.0.1:8188"):
    argv = requests.get(f"http://{host}/system_stats").json()["system"]["argv"]
    for arg in argv:
        if arg.startswith("--output-directory="):
            return arg.split("=", 1)[1]
    return None  # fall back to default
    
output_dir = get_output_dir_from_api()
print(f"Output directory: {output_dir}")