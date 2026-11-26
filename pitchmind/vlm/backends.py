"""Backends de linguagem/visão.

`ClaudeBackend`  : usa a API da Anthropic (visão + saída estruturada em JSON).
`OfflineBackend` : determinístico, sem rede — usado em testes, CI e demo.

Princípio do PitchMind: o VLM descreve o que VÊ (semântica), mas NUNCA é
fonte de números. Números vêm do tracking; o VLM recebe-os como contexto.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional, Protocol

import cv2
import numpy as np

DEFAULT_MODEL = os.getenv("PITCHMIND_MODEL", "claude-opus-5-5")


class Backend(Protocol):
    name: str

    def caption(self, image: Optional[np.ndarray], numeric_context: str) -> str: ...
    def json(self, system: str, prompt: str, schema: dict) -> dict: ...
    def text(self, system: str, prompt: str) -> str: ...


class OfflineBackend:
    name = "offline"

    def caption(self, image, numeric_context: str) -> str:
        return numeric_context

    def json(self, system, prompt, schema):  # pragma: no cover - planner offline não usa
        raise NotImplementedError

    def text(self, system, prompt):  # pragma: no cover
        raise NotImplementedError


class ClaudeBackend:
    """Requer ANTHROPIC_API_KEY (ou `ant auth login`)."""

    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def _create(self, **kw):
        resp = self.client.messages.create(model=self.model, max_tokens=4000,
                                           output_config={"effort": self.effort, **kw.pop("output_config", {})},
                                           **kw)
        if resp.stop_reason == "refusal":
            raise RuntimeError("o modelo recusou a requisição")
        return next(b.text for b in resp.content if b.type == "text")

    def caption(self, image, numeric_context: str) -> str:
        content: list[dict] = []
        if image is not None:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                            "data": base64.standard_b64encode(buf.tobytes()).decode()}})
        content.append({"type": "text", "text": (
            "Você é um analista tático de futebol. Descreva em 1-2 frases, em português, a situação "
            "tática (fase do jogo, organização, ação principal). Use os números medidos abaixo como "
            "verdade; não invente números novos.\n\nMedições do tracking:\n" + numeric_context)})
        return self._create(messages=[{"role": "user", "content": content}]).strip()

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        txt = self._create(system=system, messages=[{"role": "user", "content": prompt}],
                           output_config={"format": {"type": "json_schema", "schema": schema}})
        return json.loads(txt)

    def text(self, system: str, prompt: str) -> str:
        return self._create(system=system, messages=[{"role": "user", "content": prompt}]).strip()


def get_backend(name: str = "auto") -> Backend:
    if name == "offline":
        return OfflineBackend()
    if name == "claude":
        return ClaudeBackend()
    # auto: usa Claude se houver credencial, senão offline
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        try:
            return ClaudeBackend()
        except Exception:  # pragma: no cover
            pass
    return OfflineBackend()
