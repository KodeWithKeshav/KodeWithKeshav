import os
import math
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
from scipy.optimize import linear_sum_assignment

# --- CONFIG ---
IMG_PATH = "72bbc977-d086-4ce3-b9a0-5b253666002e.png"
GRID_W, GRID_H = 300, 340
PORTRAIT_TARGET_BANDS = 94
TRAVELER_COUNT = 900
SVG_W, SVG_H = 1180, 610
PORTRAIT_W = int(SVG_W * 0.38)

PALETTE = {
    "dark": {
        "portrait": "#A78BFA",
        "chrome": "#22D3EE",
        "accent": "#10B981",
        "bg": "#0A101F",
        "border": "#0F172A",
        "text": "#F8FAFC",
        "dim": "#94A3B8",
        "dotted": "#334155"
    },
    "light": {
        "portrait": "#7C3AED",
        "chrome": "#0891B2",
        "accent": "#10B981",
        "bg": "#0A101F",  # Explicitly locked in prompt
        "border": "#243049",
        "text": "#F8FAFC",
        "dim": "#94A3B8",
        "dotted": "#334155"
    }
}

INFO_ROWS = [
    ("Name", "Keshav S"),
    ("GitHub username", "KodeWithKeshav"),
    ("Role", "Student"),
    ("Location", "Coimbatore, IN"),
    ("Education", "B.Tech CSE"),
    ("ToolChain", "VS Code, Git, Android Studio, Figma"),
    ("Languages", "C, C++, Python, Java, Dart"),
    ("Frontend", "React, Tailwind, Bootstrap"),
    ("Backend", "Node.js, Express.js"),
    ("Database", "MongoDB, MySQL, Firebase"),
    ("Infra", "Firebase, Vercel"),
    ("LinkedIn", "linkedin.com/in/keshav-s-545345266"),
    ("Email", "kodewithkeshav@gmail.com"),
    ("Portfolio", "port-folio-kode-with-keshav.vercel.app")
]

STATUS_TEXT = "People assume the network is the machines. It's not — it's who's willing to talk to who. I just happen to speak both languages."

def segment_background(img_gray):
    arr = np.array(img_gray, dtype=float)
    threshold = np.percentile(arr, 30)
    mask = arr > threshold
    
    from scipy.ndimage import binary_closing, binary_fill_holes
    mask = binary_closing(mask, structure=np.ones((5,5)))
    mask = binary_fill_holes(mask)
    return mask

def generate_portrait_dots(mode):
    img = Image.open(IMG_PATH).convert("RGB")
    # Crop head and shoulders
    w, h = img.size
    # Better crop assuming subject is in center upper
    crop_box = (w*0.2, h*0.1, w*0.8, h*0.7)
    img = img.crop(crop_box)
    img = img.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    img_gray = img.convert("L")
    
    # Contrast 1.3x only, autocontrast(cutoff=1) + UnsharpMask(radius=3, percent=140)
    img_gray = ImageOps.autocontrast(img_gray, cutoff=1)
    img_gray = img_gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    enhancer = ImageEnhance.Contrast(img_gray)
    img_gray = enhancer.enhance(1.3)
    
    arr = np.array(img_gray, dtype=float)
    mask = segment_background(img_gray)
    
    if mode == "dark":
        # dots draw the lit subject on panel
        arr = arr
    else:
        # light mode: dots draw dark parts of photo
        arr = 255 - arr
    
    # Floyd-Steinberg Dither (serpentine)
    dots = []
    for y in range(GRID_H):
        row_iter = range(GRID_W) if y % 2 == 0 else reversed(range(GRID_W))
        for x in row_iter:
            old_pixel = arr[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            
            # hard-clear error bleed for dark mode at mask edge
            if mode == "dark" and not mask[y, x]:
                new_pixel = 0
                arr[y, x] = 0
            
            # store dot
            if new_pixel == 255:
                dots.append((x, y))
            
            quant_error = old_pixel - new_pixel
            if y % 2 == 0:
                if x + 1 < GRID_W: arr[y, x + 1] += quant_error * 7 / 16
                if y + 1 < GRID_H:
                    if x - 1 >= 0: arr[y + 1, x - 1] += quant_error * 3 / 16
                    arr[y + 1, x] += quant_error * 5 / 16
                    if x + 1 < GRID_W: arr[y + 1, x + 1] += quant_error * 1 / 16
            else:
                if x - 1 >= 0: arr[y, x - 1] += quant_error * 7 / 16
                if y + 1 < GRID_H:
                    if x + 1 < GRID_W: arr[y + 1, x + 1] += quant_error * 3 / 16
                    arr[y + 1, x] += quant_error * 5 / 16
                    if x - 1 >= 0: arr[y + 1, x - 1] += quant_error * 1 / 16

    return dots

def gen_logo_1(num_pts):
    # IoT / network (circle with nodes)
    pts = []
    cx, cy = PORTRAIT_W / 2, SVG_H / 2
    r = 80
    for _ in range(num_pts):
        angle = random.uniform(0, 2 * math.pi)
        radius = r if random.random() > 0.5 else r * 0.5
        noise_x = random.gauss(0, 5)
        noise_y = random.gauss(0, 5)
        pts.append((cx + math.cos(angle) * radius + noise_x, cy + math.sin(angle) * radius + noise_y))
    return pts

def gen_logo_2(num_pts):
    # Flutter-style abstract (angled ribbons)
    pts = []
    cx, cy = PORTRAIT_W / 2, SVG_H / 2
    for _ in range(num_pts):
        if random.random() > 0.5:
            # ribbon 1
            x = random.uniform(-40, 40)
            y = x + random.uniform(-10, 10) + 20
        else:
            # ribbon 2
            x = random.uniform(-40, 40)
            y = -x + random.uniform(-10, 10) - 20
        pts.append((cx + x, cy + y))
    return pts

def gen_logo_3(num_pts):
    # AI / neural net (layered nodes)
    pts = []
    cx, cy = PORTRAIT_W / 2, SVG_H / 2
    layers = [-50, 0, 50]
    for _ in range(num_pts):
        lx = random.choice(layers)
        ly = random.uniform(-60, 60)
        # connect visually by scattering points along lines occasionally
        if random.random() > 0.7:
            lx += random.uniform(-10, 10)
        pts.append((cx + lx, cy + ly))
    return pts

def compute_bands(dots, target_logo_centroid):
    # drift bands
    # add per-dot positional noise (sigma~4) before grouping
    noised = [(x + random.gauss(0, 4), y + random.gauss(0, 4)) for x, y in dots]
    # group into ~94 drift bands based on quantized x, y
    bands = [[] for _ in range(PORTRAIT_TARGET_BANDS)]
    for i, (nx, ny) in enumerate(noised):
        band_idx = int((nx + ny) % PORTRAIT_TARGET_BANDS)
        bands[band_idx].append(dots[i])
        
    return bands

def compute_travelers(l1, l2, l3):
    # match l1 -> l2
    dist_12 = np.linalg.norm(np.array(l1)[:, None] - np.array(l2)[None, :], axis=2)
    r_idx_12, c_idx_12 = linear_sum_assignment(dist_12)
    l2_ordered = np.array(l2)[c_idx_12]

    # match l2 -> l3
    dist_23 = np.linalg.norm(l2_ordered[:, None] - np.array(l3)[None, :], axis=2)
    r_idx_23, c_idx_23 = linear_sum_assignment(dist_23)
    l3_ordered = np.array(l3)[c_idx_23]
    
    return list(zip(l1, l2_ordered.tolist(), l3_ordered.tolist()))

def char_width(c):
    if c in "iIl1t.,- ": return 5
    if c in "MWmw": return 11
    return 8

def text_length(s, font_size):
    return sum(char_width(c) for c in s) * (font_size / 10.0)

def generate_svg(mode, out_path):
    pal = PALETTE[mode]
    
    dots = generate_portrait_dots(mode)
    
    # Scale dots to fit VISUAL.MAP area (left 38%)
    scale_x = (PORTRAIT_W - 80) / GRID_W
    scale_y = (SVG_H - 120) / GRID_H
    scale = min(scale_x, scale_y)
    ox = 40 + (PORTRAIT_W - 80 - GRID_W * scale) / 2
    oy = 80 + (SVG_H - 120 - GRID_H * scale) / 2
    
    dots_scaled = [(x * scale + ox, y * scale + oy) for x, y in dots]
    
    l1 = gen_logo_1(TRAVELER_COUNT)
    l2 = gen_logo_2(TRAVELER_COUNT)
    l3 = gen_logo_3(TRAVELER_COUNT)
    
    travelers = compute_travelers(l1, l2, l3)
    centroid_l1 = (sum(x for x, y in l1) / TRAVELER_COUNT, sum(y for x, y in l1) / TRAVELER_COUNT)
    bands = compute_bands(dots_scaled, centroid_l1)
    
    with open(out_path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}" width="{SVG_W}" height="{SVG_H}">\n')
        f.write(f'  <rect width="{SVG_W}" height="{SVG_H}" fill="{pal["bg"]}" rx="10" />\n')
        f.write(f'  <rect x="1" y="1" width="{SVG_W-2}" height="{SVG_H-2}" fill="none" stroke="{pal["border"]}" stroke-width="2" rx="10" />\n')
        
        # Top Chrome
        f.write(f'  <circle cx="25" cy="25" r="6" fill="#EF4444" />\n')
        f.write(f'  <circle cx="45" cy="25" r="6" fill="#F59E0B" />\n')
        f.write(f'  <circle cx="65" cy="25" r="6" fill="#10B981" />\n')
        f.write(f'  <text x="{SVG_W/2}" y="30" fill="{pal["text"]}" font-family="monospace" font-size="14" text-anchor="middle">profile.sh --live</text>\n')
        
        # LIVE pulsing badge
        f.write(f'  <g transform="translate({SVG_W - 220}, 15)">\n')
        f.write(f'    <rect width="50" height="20" rx="4" fill="none" stroke="#EF4444" stroke-width="1" />\n')
        f.write(f'    <circle cx="10" cy="10" r="4" fill="#EF4444">\n')
        f.write(f'      <animate attributeName="opacity" values="1;0;1" dur="1.6s" repeatCount="indefinite" />\n')
        f.write(f'    </circle>\n')
        f.write(f'    <text x="20" y="14" fill="#EF4444" font-family="monospace" font-size="12" font-weight="bold">LIVE</text>\n')
        f.write(f'  </g>\n')
        
        # Colored pill
        f.write(f'  <g transform="translate({SVG_W - 160}, 12)">\n')
        f.write(f'    <rect width="140" height="26" rx="13" fill="{pal["chrome"]}" />\n')
        f.write(f'    <text x="70" y="18" fill="{pal["bg"]}" font-family="monospace" font-size="14" font-weight="bold" text-anchor="middle">@KodeWithKeshav</text>\n')
        f.write(f'  </g>\n')
        
        # Divider
        f.write(f'  <line x1="{PORTRAIT_W}" y1="50" x2="{PORTRAIT_W}" y2="{SVG_H}" stroke="{pal["border"]}" stroke-width="2" />\n')
        f.write(f'  <line x1="0" y1="50" x2="{SVG_W}" y2="50" stroke="{pal["border"]}" stroke-width="2" />\n')
        
        # Left Panel Label
        f.write(f'  <text x="40" y="80" fill="{pal["dim"]}" font-family="monospace" font-size="13">VISUAL.MAP</text>\n')
        
        # Right Panel Label
        panel_x = PORTRAIT_W + 40
        f.write(f'  <text x="{panel_x}" y="80" fill="{pal["dim"]}" font-family="monospace" font-size="13">SYSTEM.INFO</text>\n')
        
        # Info Rows
        y_cursor = 120
        row_w = SVG_W - panel_x - 40
        for label, val in INFO_ROWS:
            f.write(f'  <text x="{panel_x}" y="{y_cursor}" fill="{pal["dim"]}" font-family="monospace" font-size="13">{label}</text>\n')
            
            # Leaders calculation
            lab_len = text_length(label, 13)
            val_len = text_length(val, 14)
            space_left = row_w - lab_len - val_len - 20
            dots_count = max(0, int(space_left / text_length(".", 13)))
            leader_str = "." * dots_count
            
            # Center leader dots properly
            f.write(f'  <text x="{panel_x + lab_len + 10}" y="{y_cursor}" fill="{pal["dotted"]}" font-family="monospace" font-size="13" textLength="{space_left}" lengthAdjust="spacingAndGlyphs">{leader_str}</text>\n')
            # Right aligned value
            f.write(f'  <text x="{panel_x + row_w - val_len}" y="{y_cursor}" fill="{pal["text"]}" font-family="monospace" font-size="14" textLength="{val_len}" lengthAdjust="spacingAndGlyphs">{val}</text>\n')
            y_cursor += 23
            
        y_cursor += 15
        
        # Status text handling (multiline)
        f.write(f'  <text x="{panel_x}" y="{y_cursor}" fill="{pal["accent"]}" font-family="monospace" font-size="13">&gt; {STATUS_TEXT[:60]}</text>\n')
        f.write(f'  <text x="{panel_x + 15}" y="{y_cursor+23}" fill="{pal["accent"]}" font-family="monospace" font-size="13">{STATUS_TEXT[60:]}</text>\n')
        
        # SMIL Animation parameters
        # Loop 14.2s. keyTimes for 3.0s (0->0.21), trans 1.3s (0.21->0.30), logo1 2.0s (0.30->0.44), trans 1.3s (0.44->0.53), 
        # logo2 2.0s (0.53->0.67), trans 1.3s (0.67->0.76), logo3 2.0s (0.76->0.90), trans to portrait 1.4s (0.90->1.0)
        loop_dur = "14.2s"
        kt = "0; 0.21; 0.30; 0.44; 0.53; 0.67; 0.76; 0.90; 1.0"
        
        # Portrait opacity keyframes: 1; 1; 0; 0; 0; 0; 0; 0; 1
        port_op = "1; 1; 0; 0; 0; 0; 0; 0; 1"
        
        # Traveler opacity: hidden during portrait phase (0;0;1;1;...;0)
        trav_op = "0; 0; 1; 1; 1; 1; 1; 1; 0"
        
        f.write(f'  <g fill="{pal["portrait"]}">\n')
        for band in bands:
            # Two nested transforms don't work cleanly in some renderers, we'll animate x and y directly or a single transform
            # The prompt says: "translate ~42% toward the first logo's centroid while fading, then returns"
            dx = (centroid_l1[0] - PORTRAIT_W/2) * 0.42
            dy = (centroid_l1[1] - SVG_H/2) * 0.42
            tx = f"0,0; 0,0; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; 0,0"
            
            path_d = "".join(f"M{x:.1f},{y:.1f}h1.5v1.5h-1.5z" for x, y in band)
            f.write(f'    <path d="{path_d}" shape-rendering="crispEdges">\n')
            f.write(f'      <animate attributeName="opacity" values="{port_op}" keyTimes="{kt}" dur="{loop_dur}" repeatCount="indefinite" />\n')
            f.write(f'      <animateTransform attributeName="transform" type="translate" values="{tx}" keyTimes="{kt}" dur="{loop_dur}" repeatCount="indefinite" />\n')
            f.write(f'    </path>\n')
        f.write(f'  </g>\n')
        
        f.write(f'  <g fill="{pal["portrait"]}">\n')
        for p1, p2, p3 in travelers:
            dx1, dy1 = p1
            dx2, dy2 = p2
            dx3, dy3 = p3
            vx = f"{dx1:.1f}; {dx1:.1f}; {dx2:.1f}; {dx2:.1f}; {dx3:.1f}; {dx3:.1f}; {dx1:.1f}; {dx1:.1f}; {dx1:.1f}"
            vy = f"{dy1:.1f}; {dy1:.1f}; {dy2:.1f}; {dy2:.1f}; {dy3:.1f}; {dy3:.1f}; {dy1:.1f}; {dy1:.1f}; {dy1:.1f}"
            
            f.write(f'    <rect width="1.5" height="1.5" rx="0.75">\n')
            f.write(f'      <animate attributeName="x" values="{vx}" keyTimes="{kt}" dur="{loop_dur}" repeatCount="indefinite" />\n')
            f.write(f'      <animate attributeName="y" values="{vy}" keyTimes="{kt}" dur="{loop_dur}" repeatCount="indefinite" />\n')
            f.write(f'      <animate attributeName="opacity" values="{trav_op}" keyTimes="{kt}" dur="{loop_dur}" repeatCount="indefinite" />\n')
            f.write(f'    </rect>\n')
        f.write(f'  </g>\n')
        
        f.write('</svg>\n')

def generate_png_test(mode, out_path):
    # Static render of first frame using pillow for validation
    pal = PALETTE[mode]
    img = Image.new("RGB", (SVG_W, SVG_H), pal["bg"])
    draw = ImageDraw.Draw(img)
    
    # draw dots
    dots = generate_portrait_dots(mode)
    scale_x = (PORTRAIT_W - 80) / GRID_W
    scale_y = (SVG_H - 120) / GRID_H
    scale = min(scale_x, scale_y)
    ox = 40 + (PORTRAIT_W - 80 - GRID_W * scale) / 2
    oy = 80 + (SVG_H - 120 - GRID_H * scale) / 2
    
    for x, y in dots:
        nx, ny = x * scale + ox, y * scale + oy
        draw.point((nx, ny), fill=pal["portrait"])
        
    img.save(out_path)

if __name__ == "__main__":
    generate_svg("dark", "dark.svg")
    generate_svg("light", "light.svg")
    generate_png_test("dark", "dark_frame1.png")
    generate_png_test("light", "light_frame1.png")
    print("Done")
