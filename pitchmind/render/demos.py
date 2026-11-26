"""Vídeos de demonstração de cada modo do PitchMind (MP4 H.264 + GIF de prévia).

    pitchmind demos --out docs/media

Modos: tracking (ByteTrack + ReID), calibration (keypoints + homografia + corte de câmera),
jersey (fusão bayesiana "18 vs 8") e radar (ao vivo).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from pitchmind.calibration.keypoints import HomographyTracker, TEMPLATE, estimate_homography
from pitchmind.identity.jersey import JerseyFuser, Observation, UNKNOWN, majority_vote
from pitchmind.models import PITCH_LENGTH as L, PITCH_WIDTH as W

BG = (23, 17, 14)
WHITE = (240, 240, 240)
YELLOW = (10, 217, 245)
BLUE = (225, 125, 47)
RED = (77, 72, 229)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------ escrita de vídeo
class VideoOut:
    """Escreve MP4 (H.264, yuv420p: toca no navegador/GitHub) via ffmpeg e gera GIF de prévia."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int = 25):
        self.path, self.size, self.fps = Path(path), size, fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg não encontrado: instale-o para gerar os vídeos")
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)

    def write(self, img: np.ndarray) -> None:
        assert img.shape[1::-1] == self.size, (img.shape, self.size)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self, gif_width: int = 640, gif_fps: int = 10) -> tuple[Path, Path]:
        self.proc.stdin.close()
        self.proc.wait()
        gif = self.path.with_suffix(".gif")
        vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.path), "-vf", vf, str(gif)], check=True)
        return self.path, gif


def _text(img, s, org, scale=0.6, color=WHITE, thick=1, bg=None):
    if bg is not None:
        (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + base + 4), bg, -1)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _header(img, title, subtitle=""):
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), BG, -1)
    _text(img, "PitchMind", (14, 29), 0.75, YELLOW, 2)
    _text(img, title, (150, 29), 0.7, WHITE, 2)
    if subtitle:
        (tw, _), _ = cv2.getTextSize(subtitle, FONT, 0.5, 1)
        _text(img, subtitle, (img.shape[1] - tw - 14, 28), 0.5, (170, 170, 170))


# ------------------------------------------------------------------ campo
def pitch_polylines() -> list[np.ndarray]:
    """Linhas do campo (em metros) como polilinhas, para desenhar sob qualquer homografia."""
    c = W / 2
    seg = lambda *p: np.array(p, np.float32)  # noqa: E731
    lines = [seg((0, 0), (L, 0), (L, W), (0, W), (0, 0)), seg((L / 2, 0), (L / 2, W))]
    for x0, s in ((0.0, 1), (L, -1)):
        lines.append(seg((x0, c - 20.16), (x0 + s * 16.5, c - 20.16), (x0 + s * 16.5, c + 20.16), (x0, c + 20.16)))
        lines.append(seg((x0, c - 9.16), (x0 + s * 5.5, c - 9.16), (x0 + s * 5.5, c + 9.16), (x0, c + 9.16)))
        a = np.linspace(-0.93, 0.93, 20)
        lines.append(np.c_[x0 + s * (11 + 9.15 * np.cos(a)), c + 9.15 * np.sin(a)].astype(np.float32)[
            np.abs(s * (11 + 9.15 * np.cos(a))) >= 16.5])
    t = np.linspace(0, 2 * np.pi, 60)
    lines.append(np.c_[L / 2 + 9.15 * np.cos(t), c + 9.15 * np.sin(t)].astype(np.float32))
    return [ln for ln in lines if len(ln) > 1]


def _proj(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def draw_pitch_view(img, H_p2i, color=WHITE, thick=2):
    for ln in pitch_polylines():
        dense = np.concatenate([np.linspace(a, b, 12) for a, b in zip(ln[:-1], ln[1:])])
        p = _proj(H_p2i, dense).astype(np.int32)
        cv2.polylines(img, [p], False, color, thick, cv2.LINE_AA)


def broadcast_camera(cx: float, span: float, size=(1280, 720), tilt: float = 0.0) -> np.ndarray:
    """H campo->imagem de uma câmera de transmissão apontada para x=cx com zoom `span` (m)."""
    w, h = size
    src = np.float32([[cx - span / 2, 0], [cx + span / 2, 0], [cx + span / 1.4, W], [cx - span / 1.4, W]])
    dst = np.float32([[w * 0.10, h * (0.20 + tilt)], [w * 0.90, h * (0.20 - tilt)], [w * 1.08, h], [w * -0.08, h]])
    return cv2.getPerspectiveTransform(src, dst)


def grass(size=(1280, 720), H_p2i=None) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (52, 120, 48)
    if H_p2i is not None:  # faixas do gramado acompanham a perspectiva
        for k in range(0, 12, 2):
            quad = _proj(H_p2i, [[k * L / 12, 0], [(k + 1) * L / 12, 0], [(k + 1) * L / 12, W], [k * L / 12, W]])
            cv2.fillPoly(img, [quad.astype(np.int32)], (58, 132, 54))
        top = _proj(H_p2i, [[-20, -6], [L + 20, -6]])
        ytop = int(max(0, min(top[:, 1].min(), h)))
        img[:ytop] = (40, 44, 48)  # arquibancada
    return img


def minimap(H_i2p, size=(1280, 720), mw=300) -> np.ndarray:
    """Mapa do campo com o trapézio que a câmera está vendo."""
    s = mw / (L + 6)
    mh = int((W + 6) * s)
    m = np.full((mh, mw, 3), (40, 96, 38), np.uint8)
    to_m = lambda p: ((np.asarray(p) + 3) * s).astype(np.int32)  # noqa: E731
    for ln in pitch_polylines():
        cv2.polylines(m, [to_m(ln)], False, (220, 220, 220), 1, cv2.LINE_AA)
    w, h = size
    if H_i2p is not None:
        fov = _proj(H_i2p, [[0, h * 0.22], [w, h * 0.22], [w, h], [0, h]])
        fov = np.clip(fov, [-3, -3], [L + 3, W + 3])
        ov = m.copy()
        cv2.fillPoly(ov, [to_m(fov)], YELLOW)
        m = cv2.addWeighted(ov, 0.30, m, 0.70, 0)
        cv2.polylines(m, [to_m(fov)], True, YELLOW, 1, cv2.LINE_AA)
    cv2.rectangle(m, (0, 0), (mw - 1, mh - 1), (200, 200, 200), 1)
    return m


# ------------------------------------------------------------------ 1. calibração
def demo_calibration(out: Path, seconds: float = 14, fps: int = 25, size=(1280, 720), seed: int = 3):
    rng = np.random.default_rng(seed)
    vo = VideoOut(out, size, fps)
    ht = HomographyTracker(frame_size=size)
    n = int(seconds * fps)
    cut = int(n * 0.55)
    n_ok = 0
    for k in range(n):
        t = k / fps
        if k < cut:   # panorâmica lenta acompanhando o jogo
            cx, span = 26 + 16 * np.sin(t * 0.45), 50 + 6 * np.sin(t * 0.3)
        else:         # corte para a câmera do outro lado, mais fechada
            cx, span = 78 + 6 * np.sin((t - cut / fps) * 0.6), 40
        Hp2i = broadcast_camera(cx, span, size)
        img = grass(size, Hp2i)
        draw_pitch_view(img, Hp2i, (235, 235, 235), 2)

        # "detector" de keypoints: pontos visíveis + ruído; 15% com id trocado (outliers)
        pts = _proj(Hp2i, TEMPLATE)
        vis = np.where((pts[:, 0] > 5) & (pts[:, 0] < size[0] - 5) & (pts[:, 1] >= size[1] * 0.19)
                       & (pts[:, 1] < size[1] - 5))[0]
        ids = vis.copy()
        obs = pts[vis] + rng.normal(0, 2.0, (len(vis), 2))
        wrong = rng.random(len(ids)) < 0.15
        ids[wrong] = rng.integers(0, len(TEMPLATE), int(wrong.sum()))
        cal = estimate_homography(obs, ids)
        H, ok = ht.update(cal)

        if cal.ok and H is not None:  # reprojeção do modelo do campo com a H ESTIMADA
            draw_pitch_view(img, np.linalg.inv(H), (255, 230, 0), 1)
        for (x, y), i, bad in zip(obs, ids, wrong):
            col = (60, 60, 235) if bad else (0, 220, 255)
            cv2.circle(img, (int(x), int(y)), 7, col, 2, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 2, col, -1)
        _header(img, "Calibração automática da câmera", "32 keypoints · RANSAC · corte de câmera")
        info = (f"keypoints {len(vis):2d} | inliers {cal.inliers:2d} | erro {cal.reproj_m:.2f} m"
                if cal.ok else f"quadro recusado ({cal.reason}): posições descartadas")
        _text(img, info, (14, size[1] - 18), 0.65, WHITE, 2, bg=BG)
        _text(img, "amarelo = detectado   vermelho = id errado (descartado pelo RANSAC)   ciano = campo reprojetado",
              (14, size[1] - 52), 0.5, WHITE, 1, bg=BG)
        if ht.cuts > 0 and k - cut < fps * 1.5 and k >= cut:
            _text(img, "CORTE DE CAMERA DETECTADO", (size[0] // 2 - 200, 100), 0.9, (20, 20, 20), 2, bg=YELLOW)
        mm = minimap(H if H is not None else None, size)
        y0, x0 = 58, size[0] - mm.shape[1] - 14
        img[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] = mm
        n_ok += int(cal.ok)
        vo.write(img)
    print(f'calibração: {n_ok}/{n} quadros calibrados')
    return vo.close()


# ------------------------------------------------------------------ 2. rastreamento
def demo_tracking(out: Path, n_frames: int = 300, fps: int = 25, size=(1280, 720), seed: int = 0):
    from pitchmind.tracking.sim import simulate_tracking
    from pitchmind.tracking.tracker import ByteTracker

    seq = simulate_tracking(n_players=16, n_frames=n_frames, size=size, seed=seed, occl_rate=0.05)
    tr = ByteTracker()
    vo = VideoOut(out, size, fps)
    trails: dict[int, list[tuple[int, int]]] = {}
    palette = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
               for h in range(0, 180, 11)]
    for k, s in enumerate(seq):
        active = tr.update(s["boxes"], s["scores"], s["frame"])
        img = s["frame"].copy()
        # detecções de baixa confiança (2º estágio do ByteTrack) em tracejado
        for b, sc in zip(s["boxes"], s["scores"]):
            if sc < 0.5:
                x1, y1, x2, y2 = b.astype(int)
                for yy in range(y1, y2, 6):
                    cv2.line(img, (x1, yy), (x1, min(yy + 3, y2)), (200, 200, 200), 1)
                    cv2.line(img, (x2, yy), (x2, min(yy + 3, y2)), (200, 200, 200), 1)
        for t in active:
            x1, y1, x2, y2 = t.box.astype(int)
            col = palette[t.track_id % len(palette)]
            foot = ((x1 + x2) // 2, y2)
            trails.setdefault(t.track_id, []).append(foot)
            pts = np.array(trails[t.track_id][-25:], np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
            cv2.ellipse(img, foot, (max((x2 - x1) // 2 + 4, 8), 6), 0, -45, 235, col, 2, cv2.LINE_AA)
            _text(img, f"#{t.track_id}", (x1 - 2, y1 - 6), 0.5, (20, 20, 20), 1, bg=col)
        lost = len(tr.lost)
        _header(img, "Rastreamento ByteTrack + ReID", "IDs estáveis em oclusões e cruzamentos")
        _text(img, f"quadro {k:3d} | tracks ativos {len(active):2d} | em espera (oclusão) {lost:2d} | "
                   f"IDs criados {tr._next_id - 1:2d} (16 jogadores)", (14, size[1] - 18), 0.6, WHITE, 2, bg=BG)
        _text(img, "tracejado = detecção de baixa confiança recuperada no 2º estágio", (14, size[1] - 52), 0.5,
              WHITE, 1, bg=BG)
        vo.write(img)
    return vo.close()


# ------------------------------------------------------------------ 3. camisas
def _jersey_crop(num: str, quality: float, hide: str | None, rng, size=(150, 190)) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), (40, 50, 200), np.uint8)
    cv2.rectangle(img, (0, int(h * 0.78)), (w, h), (30, 30, 30), -1)
    scale = 2.6
    (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_DUPLEX, scale, 6)
    org = ((w - tw) // 2, int(h * 0.30) + th)
    cv2.putText(img, num, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 6, cv2.LINE_AA)
    if hide == "left":   # braço/dobra cobrindo o primeiro dígito: "18" vira "8"
        cv2.rectangle(img, (0, int(h * 0.15)), (w // 2 - 4, int(h * 0.72)), (40, 50, 200), -1)
        cv2.line(img, (w // 2 - 6, int(h * 0.2)), (w // 2 - 20, int(h * 0.7)), (25, 35, 150), 8)
    small = max(6, int(w * (0.06 + 0.7 * quality ** 1.5)))
    img = cv2.resize(cv2.resize(img, (small, int(small * h / w))), (w, h), interpolation=cv2.INTER_NEAREST)
    k = 1 + 2 * int((1 - quality) * 6)
    img = cv2.GaussianBlur(img, (k, k), 0)
    img = np.clip(img + rng.normal(0, 18 * (1 - quality), img.shape), 0, 255).astype(np.uint8)
    return img


def demo_jersey(out: Path, fps: int = 25, size=(1280, 720), seed: int = 4):
    rng = np.random.default_rng(seed)
    roster = {"A": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "14", "16", "18", "19", "21", "23"]}
    true = "18"
    reads: list[Observation] = []
    vo = VideoOut(out, size, fps)
    n_reads, per = 26, 12   # 26 leituras ao longo da trajetória, ~0,5 s cada no vídeo
    fuser = JerseyFuser(roster)
    history = []
    for r in range(n_reads):
        hard = r < 14   # começo da jogada: jogador longe e de lado
        q = float(np.clip(rng.uniform(0.15, 0.45) if hard else rng.uniform(0.45, 0.95), 0, 1))
        hide = "left" if rng.random() < (0.65 if hard else 0.15) else None
        read = "8" if hide else ("18" if q > 0.3 else ("" if rng.random() < 0.5 else "16"))
        crop = _jersey_crop(true, q, hide, rng)
        reads.append(Observation(t=r * 0.2, read=read, conf=float(np.clip(q + 0.1, 0, 1)), quality=q))
        history.append((crop, read, q))
        post = fuser.posterior(reads, "A")
        ident = fuser.identify(1, "A", reads)
        vote = majority_vote(reads)
        for _ in range(per):
            img = np.full((size[1], size[0], 3), BG, np.uint8)
            _header(img, "Identidade: o 18 que vira 8", "fusão bayesiana ao longo da trajetória")
            # tira de recortes (últimos 7)
            _text(img, "recortes do track #7 (OCR por quadro)", (30, 82), 0.6, (170, 170, 170))
            for i, (c, rd, qq) in enumerate(history[-7:]):
                x = 30 + i * 172
                img[100:290, x:x + 150] = c
                col = (80, 200, 80) if rd == true else ((60, 60, 235) if rd else (150, 150, 150))
                _text(img, f"leu: {rd or '-'}", (x + 4, 320), 0.65, col, 2)
                cv2.rectangle(img, (x, 330), (x + int(150 * qq), 338), YELLOW, -1)
                _text(img, f"qualidade {qq:.2f}", (x + 4, 360), 0.45, (170, 170, 170))
            # decisões
            vcol = (80, 200, 80) if vote == true else (60, 60, 235)
            _text(img, "voto por maioria", (30, 430), 0.7, (170, 170, 170), 1)
            _text(img, vote if vote != UNKNOWN else "?", (30, 520), 2.6, vcol, 6)
            _text(img, f"leituras: {len(reads)}", (30, 560), 0.55, (170, 170, 170))
            fcol = (80, 200, 80) if ident.number == true else ((150, 150, 150) if ident.number == UNKNOWN else
                                                               (60, 60, 235))
            _text(img, "PitchMind (fusão)", (300, 430), 0.7, (170, 170, 170), 1)
            _text(img, ident.number if ident.number != UNKNOWN else "?", (300, 520), 2.6, fcol, 6)
            _text(img, "desconhecido: evidência insuficiente" if ident.number == UNKNOWN else
                  f"confiança {post.get(ident.number, 0):.0%}", (300, 560), 0.55, (170, 170, 170))
            # barras da posterior
            _text(img, "posterior P(número | leituras)", (640, 430), 0.6, (170, 170, 170))
            top = sorted(post.items(), key=lambda kv: -kv[1])[:5]
            for j, (num, p) in enumerate(top):
                y = 455 + j * 46
                lab = "desconh." if num == UNKNOWN else f"#{num}"
                _text(img, lab, (640, y + 22), 0.6, WHITE, 1)
                cv2.rectangle(img, (750, y + 4), (750 + int(470 * p), y + 32),
                              (80, 200, 80) if num == true else (110, 110, 110), -1)
                _text(img, f"{p:.0%}", (760 + int(470 * p), y + 26), 0.55, WHITE, 1)
            _text(img, "modelo de erro: dígito perdido (18->8), dígitos confundíveis (6/8, 1/7) e leitura espúria;"
                       " cada leitura pesa pela qualidade do recorte", (30, size[1] - 20), 0.45, (170, 170, 170))
            vo.write(img)
    for _ in range(fps * 2):
        vo.write(img)
    return vo.close()


# ------------------------------------------------------------------ 4. radar
def demo_radar(out: Path, seed: int = 7, start: float = 240, seconds: float = 80):
    """MP4 do radar ao vivo (a prévia em GIF é docs/img/live.gif, de `pitchmind animate`)."""
    import tempfile

    from pitchmind.pipeline import run_pipeline
    from pitchmind.render.animate import animate
    store, frames, _ = run_pipeline(seed=seed)
    nums = {"A": ["1", "2", "4", "3", "6", "5", "8", "10", "7", "9", "11"],
            "B": ["1", "2", "4", "3", "6", "5", "8", "10", "7", "9", "11"]}
    out = Path(out)
    with tempfile.TemporaryDirectory() as td:
        tmp = animate(frames, store.windows(), Path(td) / "radar.gif", start, seconds, numbers=nums)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "22", "-movflags", "+faststart", str(out)], check=True)
    return (out,)


def make_all(out_dir: str | Path = "docs/media") -> list[Path]:
    d = Path(out_dir)
    made = []
    for name, fn in (("tracking", demo_tracking), ("calibration", demo_calibration),
                     ("jersey", demo_jersey), ("radar", demo_radar)):
        made += list(fn(d / f"{name}.mp4"))
    return made
