import os
import sys
import csv
import time
import webbrowser
import threading
from pathlib import Path
from copy import deepcopy

from flask import Flask, render_template, jsonify, request, send_from_directory
import logging

# Disable Flask default logging for cleaner terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS & GLOBALS
# ---------------------------------------------------------------------------
BATCH_SIZE = 10
DEFAULT_FPS = 10

class AppState:
    def __init__(self):
        self.img_dir = Path('dataset/img')
        self.diffused_dir = Path('dataset/img_diffused')
        self.csv_path = Path('dataset/telemetry.csv')
        
        self.fieldnames = []
        self.rows = []
        self.frame_to_row = {}
        self.frames = []
        self.undo_stack = []

state = AppState()

# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------
def load_csv(csv_path: Path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows

def save_csv():
    with open(state.csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=state.fieldnames)
        writer.writeheader()
        writer.writerows(state.rows)

def ensure_columns():
    changed = False
    for col in ('condition', 'intersection'):
        if col not in state.fieldnames:
            state.fieldnames.append(col)
            changed = True
    for row in state.rows:
        if 'condition' not in row or row['condition'] == '':
            row['condition'] = '0'
        if 'intersection' not in row or row['intersection'] == '':
            row['intersection'] = '0'
    return changed

def init_dataset(start_idx=0):
    if not state.img_dir.exists():
        print(f"[ERROR] Image directory not found: {state.img_dir}")
        sys.exit(1)
    if not state.csv_path.exists():
        print(f"[ERROR] CSV file not found: {state.csv_path}")
        sys.exit(1)

    state.fieldnames, state.rows = load_csv(state.csv_path)
    dirty = ensure_columns()

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_imgs = sorted(p.name for p in state.img_dir.iterdir() if p.suffix.lower() in exts)

    state.frame_to_row = {row['frame']: i for i, row in enumerate(state.rows)}
    
    # Only keep images that have a CSV row
    state.frames = [f for f in all_imgs if f in state.frame_to_row]
    if start_idx > 0:
        state.frames = state.frames[start_idx:]
    
    if dirty:
        save_csv()

def get_frontend_data():
    """Returns a list of dicts with frame info to pass to the frontend."""
    data = []
    for i, frame in enumerate(state.frames):
        row_idx = state.frame_to_row.get(frame)
        row = state.rows[row_idx]
        data.append({
            'frame': frame,
            'condition': int(row.get('condition', 0)),
            'intersection': int(row.get('intersection', 0)),
            'steering': row.get('steering', 'N/A'),
            'batch_id': i // BATCH_SIZE
        })
    return data

def push_undo():
    snapshot = {
        'rows': deepcopy(state.rows),
        'frames': list(state.frames),
        'frame_to_row': dict(state.frame_to_row)
    }
    state.undo_stack.append(snapshot)
    if len(state.undo_stack) > 20:
        state.undo_stack.pop(0)

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init')
def api_init():
    return jsonify({
        'fps': DEFAULT_FPS,
        'frames': get_frontend_data()
    })

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(state.img_dir, filename)

@app.route('/api/update', methods=['POST'])
def api_update():
    data = request.json
    frame = data.get('frame')
    row_idx = state.frame_to_row.get(frame)
    
    if row_idx is not None:
        if 'condition' in data:
            state.rows[row_idx]['condition'] = str(data['condition'])
        if 'intersection' in data:
            state.rows[row_idx]['intersection'] = str(data['intersection'])
    
    return jsonify({'status': 'ok'})

@app.route('/api/delete_batch', methods=['POST'])
def api_delete_batch():
    data = request.json
    target_frame = data.get('frame')
    
    try:
        idx = state.frames.index(target_frame)
    except ValueError:
        return jsonify({'status': 'error', 'msg': 'Frame not found'}), 404
        
    batch_id = idx // BATCH_SIZE
    
    push_undo()
    
    # Gather frames in this batch
    batch_frames = [f for i, f in enumerate(state.frames) if i // BATCH_SIZE == batch_id]
    names_to_delete = set(batch_frames)
    
    # Update CSV memory
    state.rows = [r for r in state.rows if r['frame'] not in names_to_delete]
    state.frames = [f for f in state.frames if f not in names_to_delete]
    state.frame_to_row = {row['frame']: i for i, row in enumerate(state.rows)}
    
    # Delete from disk
    for f in names_to_delete:
        p = state.img_dir / f
        if p.exists():
            try: p.unlink()
            except: pass
            
        diff = state.diffused_dir / f
        if diff.exists():
            try: diff.unlink()
            except: pass
            
    save_csv()
    
    return jsonify({
        'status': 'ok',
        'frames': get_frontend_data()
    })

@app.route('/api/undo', methods=['POST'])
def api_undo():
    if not state.undo_stack:
        return jsonify({'status': 'empty'})
        
    snap = state.undo_stack.pop()
    state.rows = snap['rows']
    state.frames = snap['frames']
    state.frame_to_row = snap['frame_to_row']
    
    save_csv()
    
    return jsonify({
        'status': 'ok',
        'frames': get_frontend_data()
    })

@app.route('/api/save', methods=['POST'])
def api_save():
    save_csv()
    return jsonify({'status': 'ok'})

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_app(port=5000, start_idx=0):
    init_dataset(start_idx)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[EthosDit Web UI] Starting server at {url}")
    print("[EthosDit Web UI] Opening browser automatically...\n")
    
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index for frames')
    args = parser.parse_args()
    
    run_app(start_idx=args.start)
