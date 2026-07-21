import os
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    modified = False

    # 1. Patch VJoySender
    target_vjoy = """                if self._disable_throttle:
                    self._vjoy.data.wAxisX = self._steer
                else:
                    self._vjoy.data.wAxisX = self._steer
                    self._vjoy.data.wAxisY = y_val"""
                    
    replacement_vjoy = """                if not getattr(self, 'ai_enabled', True):
                    self._vjoy.data.wAxisX = VJOY_STEER_MID
                    self._vjoy.data.wAxisY = VJOY_SPEED_MID
                elif self._disable_throttle:
                    self._vjoy.data.wAxisX = self._steer
                else:
                    self._vjoy.data.wAxisX = self._steer
                    self._vjoy.data.wAxisY = y_val"""
                    
    if target_vjoy in content:
        content = content.replace(target_vjoy, replacement_vjoy)
        modified = True

    # 2. Patch waitKey loop to add toggle
    # Many scripts have:
    # key = cv2.waitKey(...) & 0xFF
    # if key in (ord("q"), 27):
    #     break
    # We can inject our toggle right after `if key in (ord("q"), 27):\n    break`
    
    # Let's find `key in (ord("q"), 27):` block.
    # It might look like:
    #                 if key in (ord("q"), 27):
    #                     break
    import re
    
    q_break_pattern = re.compile(r'([ \t]*)if key in \(ord\("q"\), 27\):\n([ \t]*)break\n')
    
    def q_break_repl(match):
        indent = match.group(1)
        return match.group(0) + indent + 'elif key == ord("m"):\n' + match.group(2) + 'if vjoy_sender:\n' + match.group(2) + '    vjoy_sender.ai_enabled = not getattr(vjoy_sender, "ai_enabled", True)\n' + match.group(2) + '    state = "ON" if vjoy_sender.ai_enabled else "OFF"\n' + match.group(2) + '    print(f"\\n[input] AI Control toggled to: {state}")\n'

    new_content, count = q_break_pattern.subn(q_break_repl, content)
    if count > 0 and new_content != content:
        content = new_content
        modified = True

    # Also check if it uses:
    # if key == ord('q') or key == 27:
    q_break_pattern2 = re.compile(r'([ \t]*)if key == ord\([\'"]q[\'"]\) or key == 27:\n([ \t]*)break\n')
    new_content, count = q_break_pattern2.subn(q_break_repl, content)
    if count > 0 and new_content != content:
        content = new_content
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
