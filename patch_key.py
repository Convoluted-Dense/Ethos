import os
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    modified = False

    target = 'elif key == ord("m"):'
    replacement = 'elif key == ord("0"):'
                    
    if target in content:
        content = content.replace(target, replacement)
        modified = True

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Patched {filepath}")
    else:
        print(f"No changes made to {filepath}")

if __name__ == "__main__":
    files_to_patch = [
        "test_trans.py",
        "test_v2.2.py",
        "test_cnn_v3.py",
        "test_cnn_v2.py",
        "test_cnn.py",
        "test_beamng.py",
        "predict_video_vit.py"
    ]
    for f in files_to_patch:
        if os.path.exists(f):
            patch_file(f)
