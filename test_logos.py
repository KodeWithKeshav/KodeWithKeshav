import math, random
from PIL import Image, ImageDraw

PORTRAIT_W = 448
SVG_H = 610

def gen_logo_1(num_pts):
    pts = []
    cx, cy = PORTRAIT_W / 2, SVG_H / 2
    for _ in range(num_pts):
        r = random.random()
        if r < 0.4:
            t = random.uniform(0, 2*math.pi)
            x = math.cos(t) * 70
            y = 20 + math.sin(t) * 10
        elif r < 0.9:
            t = random.uniform(math.pi, 2*math.pi)
            x = math.cos(t) * 45
            y = 20 + math.sin(t) * 50
        else:
            x = random.uniform(-45, 45)
            y = 20
        noise_x = random.gauss(0, 1)
        noise_y = random.gauss(0, 1)
        pts.append((cx + x + noise_x, cy + y - 10 + noise_y))
    return pts

def gen_logo_2(num_pts):
    pts = []
    cx, cy = PORTRAIT_W / 2, SVG_H / 2
    for _ in range(num_pts):
        r = random.random()
        if r < 0.35:
            t = random.uniform(0, 2*math.pi)
            x = math.cos(t) * 60
            y = math.sin(t) * 60
        elif r < 0.7:
            t = random.uniform(0, 2*math.pi)
            x = math.cos(t) * 40
            y = math.sin(t) * 40
        elif r < 0.85:
            x = random.uniform(-80, 80)
            y = 0
        else:
            x = 0
            y = random.uniform(-80, 80)
        noise_x = random.gauss(0, 1)
        noise_y = random.gauss(0, 1)
        pts.append((cx + x + noise_x, cy + y + noise_y))
    return pts

def gen_logo_3(num_pts):
    pts = []
    cx, cy = PORTRAIT_W / 2, SVG_H / 2
    for _ in range(num_pts):
        r = random.random()
        if r < 0.4:
            x = -20
            y = random.uniform(-50, 50)
        elif r < 0.7:
            t = random.uniform(0, 1)
            x = -20 + t * 60
            y = 0 - t * 50
        else:
            t = random.uniform(0, 1)
            x = -20 + t * 60
            y = 0 + t * 50
        noise_x = random.gauss(0, 1)
        noise_y = random.gauss(0, 1)
        pts.append((cx + x + noise_x, cy + y + noise_y))
    return pts

for i, fn in enumerate([gen_logo_1, gen_logo_2, gen_logo_3]):
    pts = fn(282)
    img = Image.new('RGB', (448, 610), 'black')
    draw = ImageDraw.Draw(img)
    for x, y in pts:
        draw.rectangle([x, y, x+1.5, y+1.5], fill='white')
    img.save(f'logo_{i+1}.png')
