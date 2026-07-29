import os
import math
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw

# --- CONFIG ---
IMG_PATH = "72bbc977-d086-4ce3-b9a0-5b253666002e.png"
GRID_W, GRID_H = 300, 340
PORTRAIT_TARGET_BANDS = 94
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
        "bg": "#0A101F",
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

def generate_dots_from_image(img, mode):
    img_gray = img.convert("L")
    img_gray = ImageOps.autocontrast(img_gray, cutoff=1)
    img_gray = img_gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    enhancer = ImageEnhance.Contrast(img_gray)
    img_gray = enhancer.enhance(1.3)
    
    arr = np.array(img_gray, dtype=float)
    mask = segment_background(img_gray)
    
    if mode == "dark":
        arr = arr
    else:
        arr = 255 - arr
    
    dots = []
    for y in range(GRID_H):
        row_iter = range(GRID_W) if y % 2 == 0 else reversed(range(GRID_W))
        for x in row_iter:
            old_pixel = arr[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            
            if mode == "dark" and not mask[y, x]:
                new_pixel = 0
                arr[y, x] = 0
            
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

def generate_portrait_dots(mode):
    img = Image.open(IMG_PATH).convert("RGB")
    w, h = img.size
    crop_box = (w*0.2, h*0.1, w*0.8, h*0.7)
    img = img.crop(crop_box)
    img = img.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    return generate_dots_from_image(img, mode)

def generate_horse_dots(mode):
    img = Image.open("horse.png").convert("RGB")
    img = ImageOps.fit(img, (GRID_W, GRID_H), Image.Resampling.LANCZOS)
    return generate_dots_from_image(img, mode)

def compute_bands(dots):
    noised = [(x + random.gauss(0, 4), y + random.gauss(0, 4)) for x, y in dots]
    bands = [[] for _ in range(PORTRAIT_TARGET_BANDS)]
    for i, (nx, ny) in enumerate(noised):
        band_idx = int((nx + ny) % PORTRAIT_TARGET_BANDS)
        bands[band_idx].append(dots[i])
    return bands

def char_width(c):
    if c in "iIl1t.,- ": return 5
    if c in "MWmw": return 11
    return 8

def text_length(s, font_size):
    return sum(char_width(c) for c in s) * (font_size / 10.0)

def generate_svg(mode, out_path):
    pal = PALETTE[mode]
    
    dots1 = generate_portrait_dots(mode)
    dots2 = generate_horse_dots(mode)
    
    scale_x = (PORTRAIT_W - 80) / GRID_W
    scale_y = (SVG_H - 120) / GRID_H
    scale = min(scale_x, scale_y)
    ox = 40 + (PORTRAIT_W - 80 - GRID_W * scale) / 2
    oy = 80 + (SVG_H - 120 - GRID_H * scale) / 2
    
    dots1_scaled = [(x * scale + ox, y * scale + oy) for x, y in dots1]
    dots2_scaled = [(x * scale + ox, y * scale + oy) for x, y in dots2]
    
    bands1 = compute_bands(dots1_scaled)
    bands2 = compute_bands(dots2_scaled)
    
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
            
            lab_len = text_length(label, 13)
            val_len = text_length(val, 14)
            space_left = row_w - lab_len - val_len - 20
            dots_count = max(0, int(space_left / text_length(".", 13)))
            leader_str = "." * dots_count
            
            f.write(f'  <text x="{panel_x + lab_len + 10}" y="{y_cursor}" fill="{pal["dotted"]}" font-family="monospace" font-size="13" textLength="{space_left}" lengthAdjust="spacingAndGlyphs">{leader_str}</text>\n')
            f.write(f'  <text x="{panel_x + row_w - val_len}" y="{y_cursor}" fill="{pal["text"]}" font-family="monospace" font-size="14" textLength="{val_len}" lengthAdjust="spacingAndGlyphs">{val}</text>\n')
            y_cursor += 23
            
        y_cursor += 15
        
        # Status text handling
        f.write(f'  <text x="{panel_x}" y="{y_cursor}" fill="{pal["accent"]}" font-family="monospace" font-size="13">&gt; {STATUS_TEXT[:60]}</text>\n')
        f.write(f'  <text x="{panel_x + 15}" y="{y_cursor+23}" fill="{pal["accent"]}" font-family="monospace" font-size="13">{STATUS_TEXT[60:]}</text>\n')
        
        loop_dur = "8s"
        kt = "0; 0.4; 0.5; 0.9; 1.0"
        port_op = "1; 1; 0; 0; 1"
        horse_op = "0; 0; 1; 1; 0"
        
        f.write(f'  <g fill="{pal["portrait"]}">\n')
        for band in bands1:
            path_d = "".join(f"M{x:.1f},{y:.1f}h1.5v1.5h-1.5z" for x, y in band)
            f.write(f'    <path d="{path_d}" shape-rendering="crispEdges">\n')
            f.write(f'      <animate attributeName="opacity" values="{port_op}" keyTimes="{kt}" dur="{loop_dur}" repeatCount="indefinite" />\n')
            f.write(f'    </path>\n')
        f.write(f'  </g>\n')
        
        f.write(f'  <g fill="{pal["portrait"]}">\n')
        for band in bands2:
            path_d = "".join(f"M{x:.1f},{y:.1f}h1.5v1.5h-1.5z" for x, y in band)
            f.write(f'    <path d="{path_d}" shape-rendering="crispEdges">\n')
            f.write(f'      <animate attributeName="opacity" values="{horse_op}" keyTimes="{kt}" dur="{loop_dur}" repeatCount="indefinite" />\n')
            f.write(f'    </path>\n')
        f.write(f'  </g>\n')
        
        f.write('</svg>\n')

def generate_png_test(mode, out_path):
    pal = PALETTE[mode]
    img = Image.new("RGB", (SVG_W, SVG_H), pal["bg"])
    draw = ImageDraw.Draw(img)
    
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
