import os
import re

with open('test_cnn_v3.py', 'r') as f:
    code = f.read()

# Add socket and struct imports
code = re.sub(
    r'(import cv2\nimport numpy as np)',
    r'\1\nimport socket\nimport struct\nimport math',
    code
)

# Insert TelemetryReceiver before VJoySender
telemetry_code = """
# ---------------------------------------------------------------------------
# UDP MotionSim telemetry receiver (BNG1)
# ---------------------------------------------------------------------------
BNG1_FORMAT = "4s" + "f" * 21
BNG1_SIZE   = struct.calcsize(BNG1_FORMAT)
BNG1_FIELDS = [
    "posX",  "posY",  "posZ",
    "velX",  "velY",  "velZ",
    "accX",  "accY",  "accZ",
    "upX",   "upY",   "upZ",
    "rollPos",  "pitchPos",  "yawPos",
    "rollVel",  "pitchVel",  "yawVel",
    "rollAcc",  "pitchAcc",  "yawAcc",
]

class TelemetryReceiver:
    def __init__(self, ip="0.0.0.0", port=4444):
        self._lock = threading.Lock()
        self._latest = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind((ip, port))
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            try:
                data, _ = self._sock.recvfrom(512)
                if len(data) >= BNG1_SIZE:
                    parts = struct.unpack_from(BNG1_FORMAT, data)
                    if parts[0][:4] == b"BNG1":
                        row = {name: parts[1 + i] for i, name in enumerate(BNG1_FIELDS)}
                        with self._lock:
                            self._latest = row
            except Exception:
                continue

    def get_speed_ms(self) -> float:
        with self._lock:
            if not self._latest:
                return 0.0
            return math.sqrt(self._latest["velX"]**2 + self._latest["velY"]**2 + self._latest["velZ"]**2)

# ---------------------------------------------------------------------------
"""

code = code.replace('# ---------------------------------------------------------------------------\n# vJoy helpers', telemetry_code + '# vJoy helpers')


# Rewrite VJoySender
old_vjoy = """class VJoySender:
    SEND_HZ = 60
    def __init__(self, vjoy_device):
        self._vjoy   = vjoy_device
        self._lock   = threading.Lock()
        self._steer  = VJOY_AXIS_MID
        self._speed  = VJOY_AXIS_MIN
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set(self, steering_raw: float, scaled_speed: float):
        with self._lock:
            self._steer = _steer_to_vjoy(steering_raw)
            self._speed = _speed_to_vjoy(scaled_speed)

    def _loop(self):
        interval = 1.0 / self.SEND_HZ
        while self._active:
            t0 = time.perf_counter()
            with self._lock:
                self._vjoy.data.wAxisX = self._steer
                self._vjoy.data.wAxisY = self._speed
            try:
                self._vjoy.update()
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            slack = interval - elapsed
            if slack > 0:
                time.sleep(slack)

    def stop(self, centre=True):
        self._active = False
        if centre:
            try:
                self._vjoy.data.wAxisX = VJOY_AXIS_MID
                self._vjoy.data.wAxisY = VJOY_AXIS_MIN
                self._vjoy.update()
            except Exception:
                pass"""

new_vjoy = """class VJoySender:
    SEND_HZ = 60
    def __init__(self, vjoy_device, telemetry: TelemetryReceiver, max_speed: float):
        self._vjoy = vjoy_device
        self._telemetry = telemetry
        self._max_speed = max_speed
        self._lock = threading.Lock()
        
        self._steer = VJOY_AXIS_MID
        self._target_speed_ms = 0.0
        
        # PID Controller state for Cruise Control
        self._integral = 0.0
        self._prev_error = 0.0
        
        # Clever Cruise Control gains
        self.Kp = 0.3    # Proportional (reacts to current error)
        self.Ki = 0.05   # Integral (overcomes hills/slopes over time)
        self.Kd = 0.1    # Derivative (dampens jerkiness)

        # Public stats for HUD
        self.current_speed_ms = 0.0
        self.pid_output = 0.0

        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set(self, steering_raw: float, scaled_speed: float):
        with self._lock:
            self._steer = _steer_to_vjoy(steering_raw)
            self._target_speed_ms = scaled_speed * self._max_speed

    def _loop(self):
        interval = 1.0 / self.SEND_HZ
        while self._active:
            t0 = time.perf_counter()
            
            # Read real speed from UDP
            cur_speed = self._telemetry.get_speed_ms()
            
            with self._lock:
                self.current_speed_ms = cur_speed
                target = self._target_speed_ms
                
                # PID calculation
                error = target - cur_speed
                self._integral += error * interval
                
                # Anti-windup for integral (prevent runaway throttle/brake memory)
                self._integral = max(-10.0, min(10.0, self._integral))
                
                derivative = (error - self._prev_error) / interval
                self._prev_error = error
                
                # Feedforward: roughly how much throttle is needed to cruise at target speed?
                # (Assuming 10% throttle per 10 m/s as a naive baseline)
                feedforward = 0.1 * (target / max(1.0, self._max_speed))
                
                # Output [-1.0 (brake) to 1.0 (throttle)]
                output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative) + feedforward
                
                # Smooth the output and clamp
                output = max(-1.0, min(1.0, output))
                self.pid_output = output
                
                # Map PID output to Y axis
                # VJOY_AXIS_MID is 0 (coast). 
                # Throttle (output > 0) -> Sweeps from MID to MIN (1)
                # Brake (output < 0) -> Sweeps from MID to MAX (32768)
                if output >= 0:
                    y_val = VJOY_AXIS_MID - int(output * (VJOY_AXIS_MID - VJOY_AXIS_MIN))
                else:
                    y_val = VJOY_AXIS_MID - int(output * (VJOY_AXIS_MAX - VJOY_AXIS_MID))
                
                y_val = max(VJOY_AXIS_MIN, min(VJOY_AXIS_MAX, y_val))
                
                self._vjoy.data.wAxisX = self._steer
                self._vjoy.data.wAxisY = y_val
                
            try:
                self._vjoy.update()
            except Exception:
                pass
                
            elapsed = time.perf_counter() - t0
            slack = interval - elapsed
            if slack > 0:
                time.sleep(slack)

    def stop(self, centre=True):
        self._active = False
        if centre:
            try:
                self._vjoy.data.wAxisX = VJOY_AXIS_MID
                self._vjoy.data.wAxisY = VJOY_AXIS_MID
                self._vjoy.update()
            except Exception:
                pass"""

code = code.replace(old_vjoy, new_vjoy)

# Update draw_hud signature and content to show PID info
old_draw_hud = """def draw_hud(img: np.ndarray, pred_steering: float, pred_speed_kmh: float, pred_offset: float, fps: float, vjoy_active: bool) -> np.ndarray:"""
new_draw_hud = """def draw_hud(img: np.ndarray, pred_steering: float, pred_speed_kmh: float, pred_offset: float, fps: float, vjoy_active: bool, cur_speed_kmh: float = 0.0, pid_out: float = 0.0) -> np.ndarray:"""
code = code.replace(old_draw_hud, new_draw_hud)

old_hud_speed = """    cv2.putText(vis, f"Speed  (pred) {pred_speed_kmh:5.1f} km/h", (10, y), _FONT, 0.6, _CLR_GREEN, 2, cv2.LINE_AA)"""
new_hud_speed = """    cv2.putText(vis, f"Target Speed  {pred_speed_kmh:5.1f} km/h", (10, y), _FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    y += 24
    cv2.putText(vis, f"Actual Speed  {cur_speed_kmh:5.1f} km/h", (10, y), _FONT, 0.6, _CLR_GREEN, 2, cv2.LINE_AA)
    y += 24
    # Draw throttle/brake bar
    cv2.putText(vis, f"PID Throttle/Brake: {pid_out:+.2f}", (10, y), _FONT, 0.6, _CLR_WHITE, 2, cv2.LINE_AA)
    bar_w = 200
    bar_x = 280
    cv2.rectangle(vis, (bar_x, y-12), (bar_x + bar_w, y+2), (60, 60, 60), -1)
    mid_x = bar_x + bar_w // 2
    if pid_out > 0:
        cv2.rectangle(vis, (mid_x, y-12), (mid_x + int(pid_out * bar_w/2), y+2), (0, 255, 0), -1)
    elif pid_out < 0:
        cv2.rectangle(vis, (mid_x + int(pid_out * bar_w/2), y-12), (mid_x, y+2), (0, 0, 255), -1)
    cv2.line(vis, (mid_x, y-14), (mid_x, y+4), (255, 255, 255), 1)"""
code = code.replace(old_hud_speed, new_hud_speed)


# Add Telemetry init in main
old_vjoy_init = """    vjoy = None
    if not opt.no_vjoy:
        try:
            import pyvjoy
            vjoy = pyvjoy.VJoyDevice(1)
            vjoy.data.wAxisX  = VJOY_AXIS_MID
            vjoy.data.wAxisY  = VJOY_AXIS_MIN
            vjoy.update()
            print("[init] vJoy device 1 acquired and centred")
        except Exception as e:
            print(f"[warn] Could not open vJoy device 1: {e}")
            vjoy = None

    vjoy_active = vjoy is not None
    vjoy_sender = None
    if vjoy_active:
        vjoy_sender = VJoySender(vjoy)"""
        
new_vjoy_init = """    telemetry = TelemetryReceiver()
    vjoy = None
    if not opt.no_vjoy:
        try:
            import pyvjoy
            vjoy = pyvjoy.VJoyDevice(1)
            vjoy.data.wAxisX  = VJOY_AXIS_MID
            vjoy.data.wAxisY  = VJOY_AXIS_MID
            vjoy.update()
            print("[init] vJoy device 1 acquired and centred")
        except Exception as e:
            print(f"[warn] Could not open vJoy device 1: {e}")
            vjoy = None

    vjoy_active = vjoy is not None
    vjoy_sender = None
    if vjoy_active:
        vjoy_sender = VJoySender(vjoy, telemetry, max_speed)"""
code = code.replace(old_vjoy_init, new_vjoy_init)

# Update draw_hud call
old_hud_call = """vis = draw_hud(raw, pred_steering, pred_speed_kmh, pred_offset, fps_display, vjoy_active)"""
new_hud_call = """cur_speed_kmh = telemetry.get_speed_ms() * 3.6
                pid_out = vjoy_sender.pid_output if vjoy_sender else 0.0
                vis = draw_hud(raw, pred_steering, pred_speed_kmh, pred_offset, fps_display, vjoy_active, cur_speed_kmh, pid_out)"""
code = code.replace(old_hud_call, new_hud_call)


with open('test_cnn_v3.py', 'w') as f:
    f.write(code)

print("test_cnn_v3.py has been updated with PID Cruise Control.")
